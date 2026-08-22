"""
RRL Feedback and Counter Update Logic
Implements the feedback aggregation outcome (y) and the soft,
confidence-weighted update step (κ) with exponential decay (γ).
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional
from .store import CandidateStore, _decay


class Attribution(str, Enum):
    """
    What the outcome was actually caused by.

    A failed task is not evidence against the retrieved document unless the document is
    what made it fail. Charging every failure to the document teaches the layer that
    good evidence is bad whenever the generator is weak, which is the fastest way to
    make reputation anti-correlated with usefulness.
    """

    RETRIEVAL = "retrieval"  # the evidence decided the outcome -> full credit/blame
    GENERATION = "generation"  # model erred despite adequate evidence -> almost none
    VERIFIER = "verifier"  # verifier itself unreliable here -> none
    PARSING = "parsing"  # output malformed before evidence mattered -> none
    TIMEOUT = "timeout"  # no usable signal -> none
    UNKNOWN = "unknown"  # unclassified -> half weight, discounted but not discarded


# Multiplier applied to the update magnitude for each attribution class. `None`
# attribution keeps full weight so existing callers are unaffected.
ATTRIBUTION_WEIGHTS: Dict[str, float] = {
    Attribution.RETRIEVAL.value: 1.0,
    Attribution.GENERATION.value: 0.05,
    Attribution.VERIFIER.value: 0.0,
    Attribution.PARSING.value: 0.0,
    Attribution.TIMEOUT.value: 0.0,
    Attribution.UNKNOWN.value: 0.5,
}


def attribution_weight(attribution: Optional[str]) -> float:
    if attribution is None:
        return 1.0
    key = attribution.value if isinstance(attribution, Attribution) else str(attribution)
    return ATTRIBUTION_WEIGHTS.get(key, 1.0)


@dataclass
class VerifierNoise:
    """
    Asymmetric error rates of the verifier producing s_gt.

    rho_fp: P(verifier says pass | answer actually wrong)
    rho_fn: P(verifier says fail | answer actually right)

    Measure these on a small audited sample rather than assuming zero. An unmodelled
    false-negative rate is indistinguishable from "this document is bad", so it biases
    reputation downward exactly where the verifier is weakest.
    """

    rho_fp: float = 0.0
    rho_fn: float = 0.0

    @property
    def is_noiseless(self) -> bool:
        return self.rho_fp <= 0.0 and self.rho_fn <= 0.0

    @property
    def separability(self) -> float:
        """1 - rho_fp - rho_fn. At or below 0 the channel carries no information."""
        return 1.0 - self.rho_fp - self.rho_fn


def correct_outcome(
    y_obs: float,
    noise: Optional[VerifierNoise] = None,
    mode: str = "bayes",
    prior: float = 0.5,
    clip: bool = True,
) -> float:
    """
    Map a noisy verifier reading to the outcome the Beta counters should be updated with.

    mode="backward"
        Unbiased estimator of the clean outcome: (y - rho_fp) / (1 - rho_fn - rho_fp).
        Unbiased in expectation but can leave [0, 1], which a Beta update cannot use
        directly; `clip` trades that unbiasedness for admissibility.
    mode="bayes"
        Posterior P(true outcome = 1 | observation), given `prior`. Always in [0, 1] and
        the natural fit for a Beta-Bernoulli posterior, at the cost of depending on the
        prior. This is the default.

    Both reduce to the identity when the verifier is noiseless.
    """
    if noise is None or noise.is_noiseless:
        return y_obs
    sep = noise.separability
    if sep <= 1e-9:
        # The channel is uninformative; fall back to the prior rather than amplifying noise.
        return prior

    if mode == "backward":
        y = (y_obs - noise.rho_fp) / sep
        return max(0.0, min(1.0, y)) if clip else y

    # Bayes posterior, interpolated for graded observations.
    p1 = (1.0 - noise.rho_fn) * prior
    p1_den = p1 + noise.rho_fp * (1.0 - prior)
    post_pass = p1 / p1_den if p1_den > 0 else prior

    p0 = noise.rho_fn * prior
    p0_den = p0 + (1.0 - noise.rho_fp) * (1.0 - prior)
    post_fail = p0 / p0_den if p0_den > 0 else prior

    w = max(0.0, min(1.0, y_obs))
    return w * post_pass + (1.0 - w) * post_fail


@dataclass
class OutcomeSignals:
    s_behave: Optional[float] = (
        None  # keep/copy capped ~0.75, minor edit ≈ 0.5, regen/rephrase/discard ≈ 0.1
    )
    s_gt: Optional[float] = (
        None  # verifier where it exists (tests pass, DB match) - overrides all if present
    )
    s_judge: Optional[float] = None  # cheap judge faithfulness-focused
    s_expl: Optional[float] = None  # thumbs up / down (1.0 / 0.0)
    # What caused this outcome. Gates how much of it lands on the document's reputation.
    attribution: Optional[Attribution] = None
    # Measured verifier error rates, if known. Left None the verifier is trusted exactly.
    noise: Optional[VerifierNoise] = None


def calculate_outcome(
    signals: OutcomeSignals,
    use_safeguards: Optional[bool] = None,
    cap_behave: bool = True,
    gt_override: bool = True,
    trust_score: float = 1.0,
) -> Optional[float]:
    """
    Computes the outcome y ∈ [0, 1].
    - Ground-truth override: If gt_override is True and s_gt is present, directly returns s_gt (no blending).
    - Asymmetry safeguard: If cap_behave is True, any positive keep/copy s_behave signal is capped at 0.75 to limit upward drift from user sycophancy.
    - Trust Score scaling: Scales s_behave and s_expl contributions by trust_score to mitigate sycophancy.
    - Otherwise, returns the weighted average of available signals:
      y = Σ w_k * s_k / Σ w_k
    """
    if use_safeguards is not None:
        cap_behave = use_safeguards
        gt_override = use_safeguards

    if gt_override and signals.s_gt is not None:
        return max(0.0, min(1.0, signals.s_gt))

    weights = {
        "s_behave": 0.45,
        "s_gt": 0.30,
        "s_judge": 0.15,
        "s_expl": 0.10,
    }

    total_weighted_sum = 0.0
    total_weight = 0.0

    for attr, weight in weights.items():
        val = getattr(signals, attr)
        if val is not None:
            # Scale user-controlled signals by trust score
            if attr in ("s_behave", "s_expl"):
                val = val * trust_score

            # Apply asymmetry safeguard for s_behave if enabled
            if attr == "s_behave" and cap_behave:
                # Capping positive signals (e.g. keep/copy > 0.5) to 0.75, keeping regens sharp at 0.10
                if val > 0.5:
                    val = min(val, 0.75)

            # Ensure signal values are clipped to [0, 1]
            val = max(0.0, min(1.0, val))
            total_weighted_sum += weight * val
            total_weight += weight

    if total_weight == 0.0:
        return None

    return total_weighted_sum / total_weight


def calculate_kappa(y: float, signals: Optional["OutcomeSignals"] = None, mode: str = "auto") -> float:
    """
    Update magnitude kappa in [0, 1].

    mode="decisiveness" (the original) uses 2*|y-0.5|, which reads a mid-range outcome as
    an uncertain one. That is wrong for a graded reward: "7 of 10 tests passed" is a
    confident measurement of partial success, not a coin flip, yet it would be applied at
    40% strength and a genuinely informative signal would be thrown away.

    mode="confidence" uses the reliability of the *channel* instead of the value: a
    verifier reading is full strength whatever it says, and a noisy verifier is
    discounted by how much of its signal survives (1 - rho_fp - rho_fn).

    mode="auto" (default) picks confidence whenever a verifier signal is present and
    falls back to decisiveness for soft behavioural signals only.
    """
    if mode == "decisiveness":
        return 2.0 * abs(y - 0.5)

    has_verifier = signals is not None and signals.s_gt is not None
    if mode == "auto" and not has_verifier:
        return 2.0 * abs(y - 0.5)

    kappa = 1.0
    if signals is not None and signals.noise is not None and not signals.noise.is_noiseless:
        kappa *= max(0.0, min(1.0, signals.noise.separability))
    return kappa


def calculate_robust_estimate(candidate, mode: str = "beta") -> float:
    """
    Computes a robust estimate of usefulness C_robust(i) from Candidate's recent outcomes.
    Supports median, trimmed (drops top 30%), mom (median of means), and beta (prior expectation) fallback.
    """
    outcomes = getattr(candidate, "recent_outcomes", [])

    # Prior expectation fallback if under 10 outcomes
    if len(outcomes) < 10:
        return candidate.alpha / (candidate.alpha + candidate.beta)

    if mode == "median":
        sorted_outcomes = sorted(outcomes)
        n = len(sorted_outcomes)
        if n % 2 == 1:
            return sorted_outcomes[n // 2]
        else:
            return (sorted_outcomes[n // 2 - 1] + sorted_outcomes[n // 2]) / 2.0

    elif mode == "trimmed":
        # Drop the top 30% of outcomes (sycophancy-bias specific)
        sorted_outcomes = sorted(outcomes)
        n = len(sorted_outcomes)
        trim_idx = int(n * 0.70)
        trimmed = sorted_outcomes[:trim_idx]
        if not trimmed:
            return sorted_outcomes[0]
        return sum(trimmed) / len(trimmed)

    elif mode == "mom":
        # Median of Means: split outcomes into 5 blocks
        n = len(outcomes)
        k = 5
        block_size = max(1, n // k)
        means = []
        for i in range(0, n, block_size):
            block = outcomes[i : i + block_size]
            if block:
                means.append(sum(block) / len(block))
        if not means:
            return 0.5
        sorted_means = sorted(means)
        m = len(sorted_means)
        if m % 2 == 1:
            return sorted_means[m // 2]
        else:
            return (sorted_means[m // 2 - 1] + sorted_means[m // 2]) / 2.0

    # Fallback to prior beta expectation
    return candidate.alpha / (candidate.alpha + candidate.beta)


def update_counters(
    store: CandidateStore,
    retrieved_sims: Dict[str, float],
    y: float,
    current_timestamp: Optional[float] = None,
    gamma: float = 0.98,
    decay_unit_sec: float = 86400.0,  # 1 day default
    credit_smoothing: float = 0.10,  # Add smoothing to avoid exploration starvation
    use_liar_counter: bool = True,
    use_adt_denoising: bool = False,
    robust_estimator_mode: str = "beta",
    signals: Optional[OutcomeSignals] = None,
    cluster_id: Optional[str] = None,
    kappa_mode: str = "auto",
    noise_correction_mode: str = "bayes",
) -> None:
    """
    Updates Beta-counters (short-term alpha, beta and permanent A, B) for candidates
    using soft, confidence-weighted updating, exponential decay, and robustness upgrades.
    """
    if not retrieved_sims:
        return

    # Calculate credit shares r(i) with smoothing
    total_smoothed_sim = sum(sim + credit_smoothing for sim in retrieved_sims.values())
    shares = {}
    if total_smoothed_sim > 0.0:
        for cid, sim in retrieved_sims.items():
            shares[cid] = (sim + credit_smoothing) / total_smoothed_sim
    else:
        share = 1.0 / len(retrieved_sims)
        for cid in retrieved_sims:
            shares[cid] = share

    update_counters_from_shares(
        store=store,
        shares=shares,
        y=y,
        current_timestamp=current_timestamp,
        gamma=gamma,
        decay_unit_sec=decay_unit_sec,
        use_liar_counter=use_liar_counter,
        use_adt_denoising=use_adt_denoising,
        robust_estimator_mode=robust_estimator_mode,
        signals=signals,
        cluster_id=cluster_id,
        kappa_mode=kappa_mode,
        noise_correction_mode=noise_correction_mode,
    )


def update_counters_from_shares(
    store: CandidateStore,
    shares: Dict[str, float],
    y: float,
    current_timestamp: Optional[float] = None,
    gamma: float = 0.98,
    decay_unit_sec: float = 86400.0,  # 1 day default
    use_liar_counter: bool = True,
    use_adt_denoising: bool = False,
    robust_estimator_mode: str = "beta",
    signals: Optional[OutcomeSignals] = None,
    cluster_id: Optional[str] = None,
    kappa_mode: str = "auto",
    noise_correction_mode: str = "bayes",
) -> None:
    """
    Updates Beta-counters using pre-computed credit shares.
    """
    if not shares:
        return

    if current_timestamp is None:
        import time

        current_timestamp = time.time()

    # Correct the reading for known verifier error rates before it becomes evidence, then
    # size the update by how much of the signal is real and how much of the outcome is
    # actually attributable to the retrieved evidence.
    if signals is not None and signals.noise is not None:
        y = correct_outcome(y, signals.noise, mode=noise_correction_mode)
    kappa = calculate_kappa(y, signals, mode=kappa_mode)
    kappa *= attribution_weight(signals.attribution if signals is not None else None)

    # Perform updates
    for cid, share_val in shares.items():
        candidate = store.get_candidate(cid)
        if not candidate:
            continue

        d_fooled = 0.0
        d_verified = 0.0

        # 1. Update liar counter if enabled and verifier signal is present
        if use_liar_counter and signals is not None and signals.s_gt is not None:
            d_verified = 1.0
            user_accepted = (signals.s_behave is not None and signals.s_behave > 0.5) or (
                signals.s_expl is not None and signals.s_expl == 1.0
            )
            verifier_failed = signals.s_gt < 0.5
            if user_accepted and verifier_failed:
                d_fooled = 1.0

        # Calculate credited outcome: scale deviation from 0.5 by credit share
        y_credited = 0.5 + share_val * (y - 0.5)

        # 2. ADT Loss downweighting if enabled
        kappa_eff = kappa
        if use_adt_denoising:
            import math

            c_robust_val = calculate_robust_estimate(candidate, robust_estimator_mode)
            loss = abs(y_credited - c_robust_val)
            kappa_eff = kappa * math.exp(-(loss**2) / 0.32)

        d_alpha = kappa_eff * share_val * y
        d_beta = kappa_eff * share_val * (1.0 - y)
        d_A = 0.25 * kappa_eff * share_val * y
        d_B = 0.25 * kappa_eff * share_val * (1.0 - y)

        if hasattr(store, "increment"):
            # SqliteCandidateStore atomic increment
            store.increment(
                candidate_id=cid,
                d_alpha=d_alpha,
                d_beta=d_beta,
                d_A=d_A,
                d_B=d_B,
                d_fooled=d_fooled,
                d_verified=d_verified,
                recent_outcome=y_credited,
                cluster_id=cluster_id,
                now=current_timestamp,
            )
        else:
            # In-memory CandidateStore updates.
            # Age the existing evidence from the last observation of any kind before
            # folding in the new one.
            anchor = candidate.decay_anchor
            dt = (current_timestamp - anchor) if anchor is not None else 0.0
            if dt > 0 and decay_unit_sec > 0:
                days = dt / decay_unit_sec
                candidate.alpha = _decay(candidate.alpha, gamma, days)
                candidate.beta = _decay(candidate.beta, gamma, days)

            # Apply updates
            candidate.alpha += d_alpha
            candidate.beta += d_beta
            candidate.A += d_A
            candidate.B += d_B
            candidate.fooled += d_fooled
            candidate.verified += d_verified
            candidate.recent_outcomes.append(y_credited)
            if len(candidate.recent_outcomes) > 30:
                candidate.recent_outcomes.pop(0)

            # Update conditional cluster counters if cluster_id is set
            if cluster_id:
                if not hasattr(candidate, "cluster_counters") or candidate.cluster_counters is None:
                    candidate.cluster_counters = {}
                if cluster_id not in candidate.cluster_counters:
                    candidate.cluster_counters[cluster_id] = {
                        "alpha": 1.0,
                        "beta": 1.0,
                        "A": 1.0,
                        "B": 1.0,
                        "fooled": 0.0,
                        "verified": 0.0,
                        "recent_outcomes": [],
                        "last_confirmed": current_timestamp,
                    }
                cc = candidate.cluster_counters[cluster_id]
                cc_anchor = cc.get("last_feedback")
                if cc_anchor is None:
                    cc_anchor = cc.get("last_confirmed", current_timestamp)
                cc_dt = (current_timestamp - cc_anchor) if cc_anchor is not None else 0.0
                if cc_dt > 0 and decay_unit_sec > 0:
                    cc_days = cc_dt / decay_unit_sec
                    cc["alpha"] = _decay(cc.get("alpha", 1.0), gamma, cc_days)
                    cc["beta"] = _decay(cc.get("beta", 1.0), gamma, cc_days)

                cc["alpha"] = cc.get("alpha", 1.0) + d_alpha
                cc["beta"] = cc.get("beta", 1.0) + d_beta
                cc["A"] = cc.get("A", 1.0) + d_A
                cc["B"] = cc.get("B", 1.0) + d_B
                cc["fooled"] = cc.get("fooled", 0.0) + d_fooled
                cc["verified"] = cc.get("verified", 0.0) + d_verified
                cc_outcomes = cc.get("recent_outcomes", [])
                cc_outcomes.append(y_credited)
                if len(cc_outcomes) > 30:
                    cc_outcomes.pop(0)
                cc["recent_outcomes"] = cc_outcomes
                if y > 0.5:
                    cc["last_confirmed"] = current_timestamp
                cc["last_feedback"] = current_timestamp
                candidate.cluster_counters[cluster_id] = cc

            # last_confirmed keeps its narrow meaning (last time this worked);
            # last_feedback advances on every observation and is the decay anchor, so
            # failures age at the same rate as successes.
            if y > 0.5:
                candidate.last_confirmed = current_timestamp
            candidate.last_feedback = current_timestamp
            candidate.last_updated = current_timestamp
            store.update_candidate(candidate)


def update_counters_with_signals(
    store: CandidateStore,
    shares: Dict[str, float],
    signals: OutcomeSignals,
    current_timestamp: Optional[float] = None,
    gamma: float = 0.98,
    decay_unit_sec: float = 86400.0,
    use_liar_counter: bool = True,
    use_adt_denoising: bool = False,
    robust_estimator_mode: str = "beta",
    cluster_id: Optional[str] = None,
    kappa_mode: str = "auto",
    noise_correction_mode: str = "bayes",
) -> None:
    """
    Updates Beta-counters for a set of candidates using pre-computed credit shares
    and a joint OutcomeSignals block, calculating candidate-specific outcomes
    based on their individual liar-counter trust scores.
    """
    if not shares:
        return

    if current_timestamp is None:
        import time

        current_timestamp = time.time()

    for cid, share_val in shares.items():
        candidate = store.get_candidate(cid)
        if not candidate:
            continue

        # 1. Calculate candidate-specific trust score
        trust_score = 1.0
        if use_liar_counter and candidate.verified > 0:
            trust_score = 1.0 - max(0.0, min(1.0, candidate.fooled / candidate.verified))

        # 2. Calculate candidate-specific outcome y
        y = calculate_outcome(signals, trust_score=trust_score)
        if y is None:
            continue
        if signals.noise is not None:
            y = correct_outcome(y, signals.noise, mode=noise_correction_mode)

        # Update magnitude: signal reliability, scaled by how much of this outcome the
        # retrieved evidence is answerable for.
        kappa = calculate_kappa(y, signals, mode=kappa_mode)
        kappa *= attribution_weight(signals.attribution)

        # Liar counter updates
        d_fooled = 0.0
        d_verified = 0.0
        if use_liar_counter and signals.s_gt is not None:
            d_verified = 1.0
            user_accepted = (signals.s_behave is not None and signals.s_behave > 0.5) or (
                signals.s_expl is not None and signals.s_expl == 1.0
            )
            verifier_failed = signals.s_gt < 0.5
            if user_accepted and verifier_failed:
                d_fooled = 1.0

        # Credited outcome
        y_credited = 0.5 + share_val * (y - 0.5)

        # ADT Loss downweighting
        kappa_eff = kappa
        if use_adt_denoising:
            import math

            c_robust_val = calculate_robust_estimate(candidate, robust_estimator_mode)
            loss = abs(y_credited - c_robust_val)
            kappa_eff = kappa * math.exp(-(loss**2) / 0.32)

        d_alpha = kappa_eff * share_val * y
        d_beta = kappa_eff * share_val * (1.0 - y)
        d_A = 0.25 * kappa_eff * share_val * y
        d_B = 0.25 * kappa_eff * share_val * (1.0 - y)

        if hasattr(store, "increment"):
            # SqliteCandidateStore atomic increment
            store.increment(
                candidate_id=cid,
                d_alpha=d_alpha,
                d_beta=d_beta,
                d_A=d_A,
                d_B=d_B,
                d_fooled=d_fooled,
                d_verified=d_verified,
                recent_outcome=y_credited,
                cluster_id=cluster_id,
                now=current_timestamp,
            )
        else:
            # Age existing evidence from the last observation of any kind.
            anchor = candidate.decay_anchor
            dt = (current_timestamp - anchor) if anchor is not None else 0.0
            if dt > 0 and decay_unit_sec > 0:
                days = dt / decay_unit_sec
                candidate.alpha = _decay(candidate.alpha, gamma, days)
                candidate.beta = _decay(candidate.beta, gamma, days)

            candidate.alpha += d_alpha
            candidate.beta += d_beta
            candidate.A += d_A
            candidate.B += d_B
            candidate.fooled += d_fooled
            candidate.verified += d_verified
            candidate.recent_outcomes.append(y_credited)
            if len(candidate.recent_outcomes) > 30:
                candidate.recent_outcomes.pop(0)

            # Update conditional cluster counters if cluster_id is set
            if cluster_id:
                if not hasattr(candidate, "cluster_counters") or candidate.cluster_counters is None:
                    candidate.cluster_counters = {}
                if cluster_id not in candidate.cluster_counters:
                    candidate.cluster_counters[cluster_id] = {
                        "alpha": 1.0,
                        "beta": 1.0,
                        "A": 1.0,
                        "B": 1.0,
                        "fooled": 0.0,
                        "verified": 0.0,
                        "recent_outcomes": [],
                        "last_confirmed": current_timestamp,
                    }
                cc = candidate.cluster_counters[cluster_id]
                cc_anchor = cc.get("last_feedback")
                if cc_anchor is None:
                    cc_anchor = cc.get("last_confirmed", current_timestamp)
                cc_dt = (current_timestamp - cc_anchor) if cc_anchor is not None else 0.0
                if cc_dt > 0 and decay_unit_sec > 0:
                    cc_days = cc_dt / decay_unit_sec
                    cc["alpha"] = _decay(cc.get("alpha", 1.0), gamma, cc_days)
                    cc["beta"] = _decay(cc.get("beta", 1.0), gamma, cc_days)

                cc["alpha"] = cc.get("alpha", 1.0) + d_alpha
                cc["beta"] = cc.get("beta", 1.0) + d_beta
                cc["A"] = cc.get("A", 1.0) + d_A
                cc["B"] = cc.get("B", 1.0) + d_B
                cc["fooled"] = cc.get("fooled", 0.0) + d_fooled
                cc["verified"] = cc.get("verified", 0.0) + d_verified
                cc_outcomes = cc.get("recent_outcomes", [])
                cc_outcomes.append(y_credited)
                if len(cc_outcomes) > 30:
                    cc_outcomes.pop(0)
                cc["recent_outcomes"] = cc_outcomes
                if y > 0.5:
                    cc["last_confirmed"] = current_timestamp
                cc["last_feedback"] = current_timestamp
                candidate.cluster_counters[cluster_id] = cc

            # last_confirmed keeps its narrow meaning (last time this worked);
            # last_feedback advances on every observation and is the decay anchor, so
            # failures age at the same rate as successes.
            if y > 0.5:
                candidate.last_confirmed = current_timestamp
            candidate.last_feedback = current_timestamp
            candidate.last_updated = current_timestamp
            store.update_candidate(candidate)


def compute_credit_shares(sims: Dict[str, float], smoothing: float = 0.10) -> Dict[str, float]:
    """
    Computes credit shares r(i) from a dictionary of similarity scores with smoothing.
    """
    if not sims:
        return {}
    total_smoothed_sim = sum(sim + smoothing for sim in sims.values())
    shares = {}
    if total_smoothed_sim > 0.0:
        for cid, sim in sims.items():
            shares[cid] = (sim + smoothing) / total_smoothed_sim
    else:
        share = 1.0 / len(sims)
        for cid in sims:
            shares[cid] = share
    return shares
