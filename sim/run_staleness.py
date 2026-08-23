"""
Staleness recovery: what happens when the right answer changes.

A static reranker scores relevance, which does not move when a document goes out of date.
A reputation layer can in principle notice and re-rank — but only if decay actually
functions, and only if failures age at the same rate as successes. Both were broken before
the clock and anchor fixes, which is why this experiment is the sharpest test of them.

Setup: the ground-truth document for each query is its own reference for the first half of
the run. At the switch point that document goes stale (it now fails) and a successor
document inside the same shortlist becomes correct. Relevance never changes, so a static
ranker cannot possibly react; only outcome feedback can.

Reported:
  * demotion lag   - observations until the stale document stops being selected
  * post-feedback  - Hit@1 in the window before vs after the switch
  * recovery       - Hit@1 on the successor by the end of the run

Usage:  python3 sim/run_staleness.py --seeds 8 --epochs 40
"""

import argparse
import os
import random
import statistics as st
import sys
from typing import Dict, List, Optional

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from rrl.store import Candidate, CandidateStore
from rrl.layer import ReputationLayer
from rrl.feedback import Attribution, OutcomeSignals, calculate_outcome, update_counters
from rrl.metrics import demotion_lag, post_feedback_performance
from run_bench import precompute, query_stream


def truth_map(pre: dict) -> Dict[int, tuple]:
    """(original, successor) per query. The successor is a real shortlist member, so the
    corrected answer is always reachable without changing the candidate set."""
    out = {}
    for tid, info in pre["per"].items():
        ranked = sorted(info["dense"], key=lambda k: -info["dense"][k])
        own = [d for d in ranked if d.startswith(f"good_{tid}_")]
        others = [d for d in ranked if not d.startswith(f"good_{tid}_")]
        if own and others:
            out[tid] = (own[0], others[0])
    return out


def run(seed: int, pre: dict, epochs: int, *, mode: str, gamma: float, noisy: Optional[tuple]):
    truth = truth_map(pre)
    problems = [p for p in pre["problems"] if p["task_id"] in truth]
    if not problems:
        return None
    contents, per = pre["contents"], pre["per"]
    random.seed(seed)
    rng = random.Random(seed * 104729)

    store = CandidateStore()
    for cid in contents:
        store.add_candidate(Candidate(id=cid, content=contents[cid], last_confirmed=0.0))
    layer = ReputationLayer(
        store,
        weights=(0.70, 0.20, 0.10, 0.10),
        gamma=gamma,
        decay_unit_sec=1.0,
        exploration_mode="ts",
    )

    stream = query_stream(seed, problems, epochs)
    n_q = len(problems)
    switch = (epochs // 2) * n_q

    hits, sel_stale, sel_by_task = [], [], {t: [] for t in truth}
    for step, p in enumerate(stream):
        tid = p["task_id"]
        original, successor = truth[tid]
        correct = original if step < switch else successor
        src = dict(per[tid]["dense"])

        if mode == "static":
            pick = max(src, key=lambda k: src[k])
        else:
            pick = layer.rescore(
                src,
                top_k=1,
                explore=True,
                now=float(step),
                cluster_id=f"q{tid}",
                epsilon=0.0,
                gamma=gamma,
                decay_unit_sec=1.0,
            ).results[0][0]

        is_right = pick == correct
        hits.append(1.0 if is_right else 0.0)
        sel_stale.append(pick)
        sel_by_task[tid].append(pick)

        if mode != "static":
            if noisy is None:
                y = 1.0 if is_right else 0.0
            else:
                p1, p2 = noisy
                y = 1.0 if rng.random() < (p1 if is_right else p2) else 0.0
            sig = OutcomeSignals(s_gt=y, attribution=Attribution.RETRIEVAL)
            update_counters(
                store,
                {pick: src[pick]},
                calculate_outcome(sig, use_safeguards=True),
                current_timestamp=float(step),
                gamma=gamma,
                decay_unit_sec=1.0,
                credit_smoothing=0.50,
                use_liar_counter=True,
                signals=sig,
                cluster_id=f"q{tid}",
            )

    # Demotion lag is measured per query in that query's own visit sequence, so it reads
    # in units of "times this query recurred" rather than global steps.
    lags = []
    for tid, picks in sel_by_task.items():
        stale = truth[tid][0]
        half = len(picks) // 2
        lag = demotion_lag(picks, stale, became_bad_at=half, consecutive=2)
        if lag is not None:
            lags.append(lag)
    pfp = post_feedback_performance(hits, switch, window=n_q * 4)
    return {
        "hit": st.mean(hits),
        "pre": pfp["before"],
        "post": pfp["after"],
        "final": st.mean(hits[-n_q * 2 :]),
        "lags": lags,
        "demoted_frac": len(lags) / max(1, len(sel_by_task)),
        "n_q": n_q,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--noisy", type=str, default="", help="p_correct,p_wrong for a noisy verifier")
    args = ap.parse_args()

    seeds = list(range(42, 42 + args.seeds))
    pre = precompute(seeds, os.path.join("/private/tmp", f"rrl_bench_{args.seeds}.pkl"))
    noisy = None
    if args.noisy:
        a, b = args.noisy.split(",")
        noisy = (float(a), float(b))

    arms = [
        ("static dense (no learning)", "static", 1.0),
        ("RRL, no decay (gamma=1.0)", "rrl", 1.0),
        ("RRL, slow decay (gamma=0.99)", "rrl", 0.99),
        ("RRL, decay (gamma=0.95)", "rrl", 0.95),
        ("RRL, fast decay (gamma=0.80)", "rrl", 0.80),
    ]

    print("=" * 102)
    print(
        f"STALENESS RECOVERY  seeds={args.seeds} epochs={args.epochs} "
        f"switch at epoch {args.epochs//2}"
        + (f"  noisy verifier {noisy}" if noisy else "  oracle verifier")
    )
    print("Relevance is unchanged by the switch, so only outcome feedback can react.")
    print("=" * 102)
    print(
        f"{'arm':<30} {'Hit@1':>7} {'pre':>7} {'post':>7} {'final':>7} "
        f"{'demoted':>8} {'lag(visits)':>12}"
    )
    print("-" * 102)
    for label, mode, gamma in arms:
        rs = [run(s, pre[s], args.epochs, mode=mode, gamma=gamma, noisy=noisy) for s in seeds]
        rs = [r for r in rs if r]
        if not rs:
            continue
        lags = [x for r in rs for x in r["lags"]]
        lag_s = f"{st.mean(lags):.1f}" if lags else "n/a"
        print(
            f"{label:<30} {st.mean(r['hit'] for r in rs)*100:6.1f}% "
            f"{st.mean(r['pre'] for r in rs)*100:6.1f}% "
            f"{st.mean(r['post'] for r in rs)*100:6.1f}% "
            f"{st.mean(r['final'] for r in rs)*100:6.1f}% "
            f"{st.mean(r['demoted_frac'] for r in rs)*100:7.0f}% {lag_s:>12}"
        )
    print("-" * 102)
    print("pre/post are the 4 epochs either side of the switch; final is the last 2 epochs.")
    print("'demoted' is the share of queries whose stale document was abandoned at all.")
    print("=" * 102)


if __name__ == "__main__":
    main()
