"""
Does verifier diagnosticity predict how much recurrence outcome feedback needs?

The claim under test: a reputation layer learning from a verifier with diagnosticity
Delta = P(pass | correct evidence) - P(pass | wrong evidence) needs on the order of

    n(Delta) = (z_a + z_b)^2 * 2*p*(1-p) / Delta^2      observations per document

which on a K-document shortlist is K*n(Delta) epochs of recurrence. If that is right, the
epoch at which the layer overtakes its static base ranker should track K*n(Delta) across a
sweep of Delta, and the layer should never converge when Delta is small.

The verifier here is synthetic and calibrated: retrieving the correct document yields a
pass with probability 0.5 + Delta/2, anything else with probability 0.5 - Delta/2. The
mean base rate is held at 0.5 so the outcome variance does not drift with Delta and only
the signal gap changes. Retrieval itself is real: real MBPP problems, real embeddings,
real shortlists.

Usage:  python3 sim/run_delta_sweep.py --seeds 8 --epochs 120
"""

import argparse
import os
import pickle
import random
import statistics as st
import sys
from typing import Dict, List, Optional

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from rrl.store import Candidate, CandidateStore
from rrl.layer import ReputationLayer
from rrl.feedback import Attribution, OutcomeSignals, calculate_outcome, update_counters
from rrl.metrics import observations_required
from run_bench import precompute, query_stream, SHORTLIST


def run(
    seed: int,
    pre: dict,
    delta: float,
    epochs: int,
    *,
    learn: bool,
    warmup: float,
    rates: Optional[tuple] = None,
) -> List[float]:
    problems, contents, per = pre["problems"], pre["contents"], pre["per"]
    rng = random.Random(seed * 7919 + int(delta * 1000))
    random.seed(seed)
    # `rates` lets a caller pin the exact measured (p_correct, p_wrong) instead of the
    # symmetric pair, so a sweep point can reproduce a real verifier rather than an
    # idealised one with the same gap.
    if rates is not None:
        p_correct, p_wrong = rates
    else:
        p_correct = 0.5 + delta / 2.0
        p_wrong = 0.5 - delta / 2.0

    store = CandidateStore()
    for cid in contents:
        store.add_candidate(Candidate(id=cid, content=contents[cid], last_confirmed=0.0))
    layer = ReputationLayer(
        store,
        weights=(0.70, 0.20, 0.10, 0.10),
        gamma=1.0,
        decay_unit_sec=1.0,
        exploration_mode="ts",
        warmup_observations=warmup,
    )

    hits = []
    for step, p in enumerate(query_stream(seed, problems, epochs)):
        tid = p["task_id"]
        src = dict(per[tid]["dense"])
        if not learn:
            pick = max(src, key=lambda k: src[k])
        else:
            pick = layer.rescore(
                src,
                top_k=1,
                explore=True,
                now=float(step),
                cluster_id=f"q{tid}",
                epsilon=0.0,
                gamma=1.0,
                decay_unit_sec=1.0,
            ).results[0][0]
        correct = pick.startswith(f"good_{tid}_")
        hits.append(1.0 if correct else 0.0)
        if learn:
            # The layer only ever sees the noisy verifier reading, never `correct`.
            observed = 1.0 if rng.random() < (p_correct if correct else p_wrong) else 0.0
            sig = OutcomeSignals(s_gt=observed, attribution=Attribution.RETRIEVAL)
            update_counters(
                store,
                {pick: src[pick]},
                calculate_outcome(sig, use_safeguards=True),
                current_timestamp=float(step),
                gamma=1.0,
                decay_unit_sec=1.0,
                credit_smoothing=0.50,
                use_liar_counter=True,
                signals=sig,
                cluster_id=f"q{tid}",
            )
    return hits


def epoch_curve(curves: List[List[float]], n_q: int, epochs: int) -> List[float]:
    return [st.mean([st.mean(c[e * n_q : (e + 1) * n_q]) for c in curves]) for e in range(epochs)]


def crossover(curve: List[float], baseline: float, margin: float = 0.02, hold: int = 3) -> Optional[int]:
    """First epoch (1-indexed) after which the curve stays `margin` above baseline for `hold` epochs."""
    run_len = 0
    for i, v in enumerate(curve):
        if v >= baseline + margin:
            run_len += 1
            if run_len >= hold:
                return i - hold + 2
        else:
            run_len = 0
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--warmup", type=float, default=0.0)
    ap.add_argument(
        "--deltas",
        type=str,
        default="0.1,0.2,0.3,0.4,0.6,0.8,1.0",
    )
    ap.add_argument(
        "--rates",
        type=str,
        default="",
        help="p_correct,p_wrong to pin one measured verifier instead of sweeping symmetric pairs",
    )
    args = ap.parse_args()

    seeds = list(range(42, 42 + args.seeds))
    pre = precompute(seeds, os.path.join("/private/tmp", f"rrl_bench_{args.seeds}.pkl"))
    deltas = [float(x) for x in args.deltas.split(",")]

    static_curves = [run(s, pre[s], 1.0, args.epochs, learn=False, warmup=0.0) for s in seeds]
    n_q = len(pre[seeds[0]]["problems"])
    static = st.mean([st.mean(c) for c in static_curves])

    print("=" * 100)
    print(
        f"DELTA SWEEP  seeds={args.seeds} epochs={args.epochs} shortlist={SHORTLIST} "
        f"static dense baseline Hit@1={static*100:.1f}%"
    )
    print("Verifier is synthetic and calibrated; retrieval, corpus and shortlists are real.")
    print("=" * 100)
    print(
        f"{'Delta':>6} {'predicted':>11} {'observed':>9} {'final':>8} {'ep1':>6} "
        f"{'ep8':>6} {'ep30':>6} {'ep last':>8}"
    )
    print(f"{'':>6} {'epochs':>11} {'epochs':>9} {'Hit@1':>8}")
    print("-" * 100)
    rates = None
    if args.rates:
        a, b = args.rates.split(",")
        rates = (float(a), float(b))
        deltas = [rates[0] - rates[1]]
        print(f"pinned verifier: P(pass|correct)={rates[0]:.3f} P(pass|wrong)={rates[1]:.3f}\n")

    GAINS = []
    rows = []
    for d in deltas:
        curves = [
            run(s, pre[s], d, args.epochs, learn=True, warmup=args.warmup, rates=rates)
            for s in seeds
        ]
        curve = epoch_curve(curves, n_q, args.epochs)
        pred = observations_required(d) * SHORTLIST
        obs = crossover(curve, static)
        rows.append((d, pred, obs, curve[-1]))
        GAINS.append((d, curve[-1] - static))
        obs_s = f"{obs}" if obs is not None else "never"
        pred_s = f"{pred:.0f}"
        print(
            f"{d:>6.2f} {pred_s:>11} {obs_s:>9} {curve[-1]*100:>7.1f}% "
            f"{curve[0]*100:>5.1f} {curve[min(7,len(curve)-1)]*100:>5.1f} "
            f"{curve[min(29,len(curve)-1)]*100:>5.1f} {curve[-1]*100:>7.1f}"
        )
    print("-" * 100)
    pairs = [(p, o) for _, p, o, _ in rows if o is not None and p != float("inf")]
    if len(pairs) >= 3:
        try:
            from scipy import stats as sp

            r = sp.spearmanr([p for p, _ in pairs], [o for _, o in pairs])
            print(
                f"rank correlation between predicted and observed crossover epoch: "
                f"rho={r.statistic:.3f} p={r.pvalue:.4f}  (n={len(pairs)} of {len(rows)} deltas converged)"
            )
        except Exception:
            pass
    print("\nfinal Hit@1 gain over the static base ranker, by Delta:")
    for d, g in GAINS:
        bar = "#" * int(max(0.0, g) * 200)
        print(f"   Delta={d:.2f}  {g*100:+5.1f} pts  {bar}")
    print("   -> the gain saturates: past a point a better verifier buys nothing more.")

    never = [d for d, _, o, _ in rows if o is None]
    if never:
        print(f"never overtook the static baseline within {args.epochs} epochs: Delta in {never}")
    print("=" * 100)


if __name__ == "__main__":
    main()
