"""
RRL Reputation Layer
Implements the retriever-agnostic ReputationLayer, which computes RRL scores over
similarity dictionary outputs and updates document reputations based on outcomes.
"""

from dataclasses import dataclass
import random
import uuid
from typing import Dict, List, Optional, Tuple

from .clock import Clock, resolve_now
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
        exploration_mode: str = "ts",
        warmup_observations: float = 0.0,
        shrinkage_threshold: float = 10.0,
        clock: Optional[Clock] = None,
        strict_weights: bool = True,
    ):
        self.store = store
        self.weights = weights
        self.robust_estimator_mode = robust_estimator_mode
        self.use_optimistic_prior = use_optimistic_prior
        self.gamma = gamma
        self.decay_unit_sec = decay_unit_sec
        self.use_clustering = use_clustering
        self.exploration_mode = exploration_mode
        self.warmup_observations = warmup_observations
        self.shrinkage_threshold = shrinkage_threshold
        self.clock = clock

        # An exploration term wider than the relevance term it perturbs makes the top
        # rank mostly noise, and with a Beta(1,1) posterior it never narrows. Warn loudly
        # rather than silently producing a randomised ranker.
        w_sim, _, _, w_explore = weights
        if strict_weights and w_sim > 0.0 and w_explore > 0.5 * w_sim:
            import warnings

            warnings.warn(
                f"w_explore={w_explore} exceeds half of w_sim={w_sim}; exploration will "
                f"dominate relevance in the ranking. Prefer w_explore <= w_sim/2.",
                RuntimeWarning,
                stacklevel=2,
            )

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
        shortlist_k: Optional[int] = None,
        exploration_mode: Optional[str] = None,
        warmup_observations: Optional[float] = None,
    ) -> RescoreResult:
        """
        sims: candidate_id -> normalized relevance from ANY retriever.
        Returns ranked ids + scores + a response_id with frozen credit shares.

        shortlist_k
            Score only the top `shortlist_k` candidates by incoming relevance. This is
            what makes the layer a *reranker*: the reputation signal reorders a
            shortlist the base retriever already vouched for, instead of competing with
            relevance across the whole corpus. Leave None to score everything passed in.
        exploration_mode
            "ts"     - Thompson sampling on the posterior alone (recommended).
            "ucb"    - deterministic upper confidence bound.
            "legacy" - the similarity-scaled sample-plus-rarity term kept for
                       reproducing older runs; its range can exceed the relevance term.
        warmup_observations
            Force uniform exploration among the least-observed shortlist entries until
            every entry has this much evidence. A Beta(1,1) posterior is indistinguishable
            from noise, so ranking on it before any evidence exists is worse than
            sampling deliberately.
        """
        w_sim, w_c, w_p, w_explore = (
            override_weights if override_weights is not None else self.weights
        )
        expl_mode = exploration_mode if exploration_mode is not None else self.exploration_mode
        warmup = (
            warmup_observations if warmup_observations is not None else self.warmup_observations
        )
        robust_mode = (
            robust_estimator_mode
            if robust_estimator_mode is not None
            else self.robust_estimator_mode
        )
        decay_gamma = gamma if gamma is not None else self.gamma
        decay_sec = decay_unit_sec if decay_unit_sec is not None else self.decay_unit_sec
        now = resolve_now(self.clock, now)

        # Query clustering setup
        eff_cluster_id = None
        if cluster_id is not None:
            eff_cluster_id = cluster_id
        elif self.use_clustering:
            eff_cluster_id = "cluster_0"

        # Two-stage gate: keep only the base retriever's best `shortlist_k` candidates so
        # reputation reranks vetted relevance rather than overriding it corpus-wide.
        if shortlist_k is not None and shortlist_k > 0 and len(sims) > shortlist_k:
            keep = sorted(sims, key=lambda k: sims[k], reverse=True)[:shortlist_k]
            sims = {k: sims[k] for k in keep}

        ranked_candidates = []
        for cid, sim in sims.items():
            candidate = self.store.get_candidate(cid)
            if not candidate:
                self.track(cid)
                candidate = self.store.get_candidate(cid)
                if not candidate:
                    continue

            # Age the counters for scoring WITHOUT writing them back. The SQL store
            # already decays on read inside the query; the in-memory store computes an
            # effective view here. Either way the stored state only moves on feedback,
            # so scoring the same candidate repeatedly is idempotent.
            if hasattr(self.store, "increment"):
                alpha_global = candidate.alpha
                beta_global = candidate.beta
                A_global = candidate.A
                B_global = candidate.B
            else:
                alpha_global, beta_global, A_global, B_global = candidate.effective_counters(
                    now=now, gamma=decay_gamma, decay_unit_sec=decay_sec
                )

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
                # Same anchor rule as the global counters: age from the last observation
                # of any kind, and skip decay entirely when there is none.
                cc_anchor = cc.get("last_feedback")
                if cc_anchor is None:
                    cc_anchor = cc.get("last_confirmed")

                cc_dt = 0.0
                if cc_anchor is not None and now is not None and decay_sec > 0:
                    cc_dt = (now - cc_anchor) / decay_sec

                alpha_c = _decay(cc.get("alpha", 1.0), decay_gamma, cc_dt)
                beta_c = _decay(cc.get("beta", 1.0), decay_gamma, cc_dt)
                A_c = cc.get("A", 1.0)
                B_c = cc.get("B", 1.0)
                # Evidence mass on the short-term cluster counters drives shrinkage. The
                # permanent A/B counters accumulate at a quarter rate, so reading n from
                # them understates how much the cluster has actually seen.
                n_cluster = max(0.0, alpha_c + beta_c - 2.0)

            # Hierarchical shrinkage: a query-conditional estimate with little evidence
            # is pulled toward the pooled global estimate, and takes over as its own
            # evidence accumulates. This is the sample-efficiency lever — per-query
            # counters alone are unbiased but starved, global counters alone are dense
            # but blind to the query.
            lam = min(1.0, max(0.0, n_cluster / self.shrinkage_threshold))

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
                if expl_mode == "ts":
                    # Pure Thompson sampling: replace the point estimate of usefulness
                    # with a draw from its posterior. Uncertainty then enters the score
                    # through the same channel and with the same scale as the estimate
                    # it perturbs, so exploration cannot outvote relevance.
                    score += w_explore * (random.betavariate(alpha_val, beta_val) - C_robust)
                elif expl_mode == "ucb":
                    score += w_explore * rarity_bonus
                else:
                    # Legacy term, retained to reproduce pre-fix runs. Its range is
                    # w_explore * sim * (1 + rarity), which exceeds the relevance term
                    # whenever w_explore approaches w_sim.
                    ts_sample = random.betavariate(alpha_val, beta_val)
                    score += w_explore * sim * (ts_sample + rarity_bonus)

            ranked_candidates.append((candidate.id, score, sim))

        # Sort by total score descending
        ranked_candidates.sort(key=lambda x: x[1], reverse=True)

        # Warm-up phase: until every shortlist entry carries `warmup` evidence, spend the
        # top slot on the least-observed candidate instead of ranking on posteriors that
        # are still indistinguishable from the prior.
        if explore and warmup and warmup > 0.0 and ranked_candidates:
            obs = {}
            for cid, _, _ in ranked_candidates:
                c = self.store.get_candidate(cid)
                obs[cid] = c.observations() if c is not None else 0.0
            if obs and min(obs.values()) < warmup:
                fewest = min(obs.values())
                pool = [t for t in ranked_candidates if obs[t[0]] <= fewest + 1e-9]
                chosen = random.choice(pool)
                ranked_candidates = [chosen] + [t for t in ranked_candidates if t[0] != chosen[0]]

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
