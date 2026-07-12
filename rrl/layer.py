"""
RRL Reputation Layer
Implements the retriever-agnostic ReputationLayer, which computes RRL scores over
similarity dictionary outputs and updates document reputations based on outcomes.
"""

from dataclasses import dataclass
import random
import uuid
from typing import Dict, List, Optional, Tuple, Any

from .store import Candidate, CandidateStore, _decay
from .feedback import (
    OutcomeSignals,
    calculate_robust_estimate,
    update_counters_with_signals,
    compute_credit_shares,
)


@dataclass
class RescoreResult:
    results: List[Tuple[str, float, float]]  # List of (candidate_id, total_score, sim_score)
    response_id: str


class ReputationLayer:
    def __init__(
        self,
        store: CandidateStore,
        *,
        weights: Tuple[float, float, float, float] = (0.55, 0.30, 0.15, 0.0),
        robust_estimator_mode: str = "beta",
        use_optimistic_prior: bool = True,
        gamma: float = 0.98,
        decay_unit_sec: float = 86400.0,
        use_clustering: bool = False,
    ):
        self.store = store
        self.weights = weights
        self.robust_estimator_mode = robust_estimator_mode
        self.use_optimistic_prior = use_optimistic_prior
        self.gamma = gamma
        self.decay_unit_sec = decay_unit_sec
        self.use_clustering = use_clustering

    def track(self, id: str, content: str = "", metadata: Optional[dict] = None) -> None:
        """
        Register a document the layer should keep reputation for (upsert).

        Note: Registering unknown/unrecognized IDs will write empty-content rows.
        This is intended for on-the-fly tracking of external documents that have not
        been fully ingested yet.
        """
        candidate = self.store.get_candidate(id)
        if candidate:
            if content:
                candidate.content = content
            if metadata:
                candidate.metadata.update(metadata)
            self.store.update_candidate(candidate)
        else:
            candidate = Candidate(id=id, content=content, metadata=metadata or {})
            self.store.add_candidate(candidate)


    def rescore(
        self,
        sims: Dict[str, float],
        *,
        top_k: int = 5,
        explore: bool = True,
        now: Optional[float] = None,
        cluster_id: Optional[str] = None,
        override_weights: Optional[Tuple[float, float, float, float]] = None,
        epsilon: float = 0.15,
        robust_estimator_mode: Optional[str] = None,
        gamma: Optional[float] = None,
        decay_unit_sec: Optional[float] = None,
    ) -> RescoreResult:
        """
        sims: candidate_id -> normalized relevance from ANY retriever.
        Returns ranked ids + scores + a response_id with frozen credit shares.
        """
        w_sim, w_c, w_p, w_explore = (
            override_weights if override_weights is not None else self.weights
        )
        robust_mode = (
            robust_estimator_mode
            if robust_estimator_mode is not None
            else self.robust_estimator_mode
        )
        decay_gamma = gamma if gamma is not None else self.gamma
        decay_sec = decay_unit_sec if decay_unit_sec is not None else self.decay_unit_sec

        # Query clustering setup
        eff_cluster_id = None
        if cluster_id is not None:
            eff_cluster_id = cluster_id
        elif self.use_clustering:
            eff_cluster_id = "cluster_0"

        ranked_candidates = []
        for cid, sim in sims.items():
            candidate = self.store.get_candidate(cid)
            if not candidate:
                self.track(cid)
                candidate = self.store.get_candidate(cid)
                if not candidate:
                    continue

            # Decay on read if not handled by SQL store (i.e. for in-memory)
            alpha_global = candidate.alpha
            beta_global = candidate.beta
            A_global = candidate.A
            B_global = candidate.B

            if not hasattr(self.store, "increment"):
                last_confirmed = candidate.last_confirmed
                dt = now - last_confirmed if now is not None else 0.0
                if dt > 0 and decay_sec > 0:
                    days = dt / decay_sec
                    candidate.alpha = _decay(candidate.alpha, decay_gamma, days)
                    candidate.beta = _decay(candidate.beta, decay_gamma, days)
                    candidate.last_updated = now if now is not None else candidate.last_updated
                alpha_global = candidate.alpha
                beta_global = candidate.beta


            # Apply optimistic prior for cold-start / new docs
            if self.use_optimistic_prior and (
                A_global + B_global <= 2.0 or (alpha_global == 1.0 and beta_global == 1.0)
            ):
                alpha_global = 2.0

            # Hierarchical query-conditional cluster counters
            alpha_c = 1.0
            beta_c = 1.0
            A_c = 1.0
            B_c = 1.0
            n_cluster = 0.0

            cluster_counters = getattr(candidate, "cluster_counters", {})
            if eff_cluster_id and eff_cluster_id in cluster_counters:
                cc = cluster_counters[eff_cluster_id]
                cc_lc = cc.get("last_confirmed", candidate.last_confirmed)

                cc_dt = 0.0
                if now is not None and decay_sec > 0:
                    cc_dt = (now - cc_lc) / decay_sec

                alpha_c = _decay(cc.get("alpha", 1.0), decay_gamma, cc_dt)
                beta_c = _decay(cc.get("beta", 1.0), decay_gamma, cc_dt)
                A_c = cc.get("A", 1.0)
                B_c = cc.get("B", 1.0)
                n_cluster = max(0.0, A_c + B_c - 2.0)

            # Shrinkage interpolation (K_threshold = 10.0)
            N_threshold = 10.0
            lam = min(1.0, max(0.0, n_cluster / N_threshold))

            alpha = (1.0 - lam) * alpha_global + lam * alpha_c
            beta = (1.0 - lam) * beta_global + lam * beta_c
            A = (1.0 - lam) * A_global + lam * A_c
            B = (1.0 - lam) * B_global + lam * B_c

            # Robust estimation for exploitation C_robust
            C_robust = calculate_robust_estimate(candidate, robust_mode)
            if robust_mode == "beta":
                C_robust = alpha / (alpha + beta)

            # P(i) = A(i) / (A(i) + B(i))
            P_i = A / (A + B) if (A + B) > 0 else 0.5

            # Rarity/uncertainty UCB bonus with exploration floor (min 0.05)
            rarity_bonus = max(0.05, 1.0 / ((alpha + beta) ** 0.5))

            # Calculate total score using configured weights
            score = w_sim * sim + w_c * C_robust + w_p * P_i
            if explore:
                alpha_val = max(1e-5, alpha)
                beta_val = max(1e-5, beta)
                ts_sample = random.betavariate(alpha_val, beta_val)
                # Scale exploration by similarity score to prevent exploring completely irrelevant candidates
                score += w_explore * sim * (ts_sample + rarity_bonus)

            ranked_candidates.append((candidate.id, score, sim))

        # Sort by total score descending
        ranked_candidates.sort(key=lambda x: x[1], reverse=True)
        results = ranked_candidates[:top_k]

        # Epsilon-greedy exploration over the full candidate set
        if explore and random.random() < epsilon and len(sims) > top_k:
            all_cids = list(sims.keys())
            selected_cids = [r[0] for r in results[: top_k - 1]]
            raw_pool = [
                self.store.get_candidate(cid) for cid in all_cids if cid not in selected_cids
            ]
            candidate_pool = [c for c in raw_pool if c is not None]

            if candidate_pool:
                min_count = min(c.alpha + c.beta for c in candidate_pool)
                least_explored = [
                    c for c in candidate_pool if (c.alpha + c.beta) <= min_count + 1e-5
                ]
                explorer_cand = random.choice(least_explored)

                explorer_tuple = None
                for r_tuple in ranked_candidates:
                    if r_tuple[0] == explorer_cand.id:
                        explorer_tuple = r_tuple
                        break

                if explorer_tuple:
                    results = results[: top_k - 1] + [explorer_tuple]

        # Compute credit shares and save pending
        retrieved_sims = {cid: sim for cid, _, sim in results}
        shares = compute_credit_shares(retrieved_sims, smoothing=0.10)
        response_id = str(uuid.uuid4())
        if shares:
            self.store.save_pending(response_id, shares, eff_cluster_id, now)

        return RescoreResult(results=results, response_id=response_id)

    def record_feedback(
        self,
        response_id: str,
        *,
        s_behave: Optional[float] = None,
        s_gt: Optional[float] = None,
        s_judge: Optional[float] = None,
        s_expl: Optional[float] = None,
        now: Optional[float] = None,
    ) -> Dict[str, float]:
        res = self.store.pop_pending(response_id)
        if res is None:
            raise KeyError(f"Pending shares for response_id {response_id} not found.")
        shares, cluster_id = res
        signals = OutcomeSignals(
            s_behave=s_behave,
            s_gt=s_gt,
            s_judge=s_judge,
            s_expl=s_expl,
        )
        update_counters_with_signals(
            store=self.store,
            shares=shares,
            signals=signals,
            current_timestamp=now,
            gamma=self.gamma,
            decay_unit_sec=self.decay_unit_sec,
            cluster_id=cluster_id,
            robust_estimator_mode=self.robust_estimator_mode,
        )
        return shares

