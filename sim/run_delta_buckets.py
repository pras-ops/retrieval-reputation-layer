"""
Phase 3: does measured per-task diagnosticity predict how much RRL gains, on real feedback?

This is the experiment the earlier work could not run. It needs three things at once:

  * a per-task Delta estimated from real generations, which requires temperature > 0 and
    several samples per (task, document) pair -- at temperature 0 a per-task rate is its own
    single sample and the distribution collapses onto 0 and 1;
  * a real verifier in the feedback loop, not a calibrated synthetic channel;
  * a sample split, so Delta is estimated on different generations than RRL is scored on.

The split is not optional. Bucketing tasks by Delta and then scoring RRL on the same samples
is selection on the dependent variable: each high-Delta bucket would carry its own selection
noise into the result and inflate the gain. Samples 0..k/2-1 estimate Delta; samples k/2..k-1
supply the outcomes RRL learns from and is measured on.

Usage:
  python3 sim/run_delta_buckets.py --probe delta_probe_nothink.jsonl --epochs 64
"""

import argparse
import json
import math
import os
import random
import statistics as st
import sys
from collections import defaultdict

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from rrl.store import Candidate, CandidateStore
from rrl.layer import ReputationLayer
from rrl.feedback import Attribution, OutcomeSignals, calculate_outcome, update_counters
from rrl.metrics import diagnosticity, reciprocal_rank, ndcg_at_k
from run_bench_v2 import precompute, ARMS

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PROBE_DIR = os.path.join(ROOT, "data", "outcomes", "probe")


def load_probe(fname):
    rows = []
    for line in open(os.path.join(PROBE_DIR, fname)):
        if not line.strip():
            continue
        d = json.loads(line)
        if "error" not in d:
            rows.append(d)
    return rows


def split_samples(rows, k):
    """(estimate, evaluate) keyed by (task_id, doc_id) -> list of binary outcomes."""
    half = k // 2
    est, ev = defaultdict(list), defaultdict(list)
    for r in rows:
        key = (r["task_id"], r["doc_id"])
        (est if r["sample_idx"] < half else ev)[key].append(float(r["passed"]))
    return est, ev


def per_task_delta(est, tasks_by_id):
    out = {}
    for tid, t in tasks_by_id.items():
        correct = t["correct_doc_id"]
        c = est.get((tid, correct), [])
        w = [v for d in t["candidate_doc_ids"] if d != correct for v in est.get((tid, d), [])]
        if not c or not w:
            continue
        out[tid] = {
            "p_correct": st.mean(c),
            "p_wrong": st.mean(w),
            "delta": st.mean(c) - st.mean(w),
            "n_correct": len(c),
            "n_wrong": len(w),
        }
    return out


def bucket_of(d):
    if d <= 0.05:
        return "none (<=0.05)"
    if d < 0.30:
        return "low (0.05-0.30)"
    if d < 0.60:
        return "medium (0.30-0.60)"
    return "high (>=0.60)"


def run(seed, pre, arm, task_ids, ev, epochs, use_oracle=False):
    tasks = [t for t in pre["tasks"] if t["task_id"] in task_ids]
    if not tasks:
        return None
    per = pre["per"]
    random.seed(seed)
    draw = random.Random(seed * 7919)

    rep = arm.get("rep")
    layer = None
    if rep is not None:
        store = CandidateStore()
        for c in pre["doc_ids"]:
            store.add_candidate(Candidate(id=c, content="", last_confirmed=0.0))
        layer = ReputationLayer(
            store, weights=rep["w"], gamma=1.0, decay_unit_sec=1.0, exploration_mode="ts"
        )

    order = random.Random(seed)
    stream = []
    for _ in range(epochs):
        o = list(tasks)
        order.shuffle(o)
        stream.extend(o)

    hits, solved, no_signal = [], [], 0
    for step, t in enumerate(stream):
        src = dict(per[t["task_id"]][arm["base"]])
        correct = t["correct_doc_id"]
        if layer is None:
            pick = max(src, key=lambda k: src[k])
        else:
            cl = t["task_id"] if rep.get("cond") else None
            pick = layer.rescore(
                src, top_k=1, explore=True, now=float(step), cluster_id=cl,
                epsilon=0.0, gamma=1.0, decay_unit_sec=1.0,
            ).results[0][0]
        ok = pick == correct
        hits.append(1.0 if ok else 0.0)

        # Outcome comes from the held-out generations for this exact (task, document) pair.
        pool = ev.get((t["task_id"], pick), [])
        if pool:
            y_real = pool[draw.randrange(len(pool))]
            solved.append(y_real)
        else:
            no_signal += 1
            y_real = None

        if layer is not None:
            y = (1.0 if ok else 0.0) if use_oracle else y_real
            if y is None:
                continue
            sig = OutcomeSignals(s_gt=y, attribution=Attribution.RETRIEVAL)
            update_counters(
                layer.store, {pick: src[pick]},
                calculate_outcome(sig, use_safeguards=True),
                current_timestamp=float(step), gamma=1.0, decay_unit_sec=1.0,
                credit_smoothing=0.50, use_liar_counter=True, signals=sig,
                cluster_id=(t["task_id"] if rep.get("cond") else None),
            )
    return {
        "hit": hits, "solved": solved, "n_q": len(tasks),
        "coverage": 1.0 - no_signal / max(1, len(hits)),
    }


def paired(a, b):
    d = [x - y for x, y in zip(a, b)]
    if len(d) < 2:
        return st.mean(d) if d else 0.0, float("nan")
    m, sem = st.mean(d), st.stdev(d) / math.sqrt(len(d))
    if sem == 0:
        return m, float("nan")
    try:
        from scipy import stats as sp
        return m, float(2 * (1 - sp.t.cdf(abs(m / sem), len(d) - 1)))
    except Exception:
        return m, float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", default="delta_probe_nothink.jsonl")
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=64)
    ap.add_argument("--arm", default="dense + RRL + pooling")
    ap.add_argument("--baseline", default="dense only")
    args = ap.parse_args()

    pre = precompute()
    tasks_by_id = {t["task_id"]: t for t in pre["tasks"]}
    rows = load_probe(args.probe)
    est, ev = split_samples(rows, args.samples)
    deltas = per_task_delta(est, tasks_by_id)
    seeds = list(range(42, 42 + args.seeds))

    thinking = {r.get("thinking_budget") for r in rows}
    print("=" * 100)
    print(f"DELTA BUCKETS on real feedback   probe={args.probe}  rows={len(rows):,}")
    print(f"tasks with a Delta estimate: {len(deltas)}   thinking_budget={thinking}")
    print(f"samples 0-{args.samples//2-1} estimate Delta | samples {args.samples//2}-{args.samples-1} supply outcomes")
    print("=" * 100)

    allc = [v for k, l in est.items() if k[1] == tasks_by_id[k[0]]["correct_doc_id"] for v in l]
    allw = [v for k, l in est.items() if k[1] != tasks_by_id[k[0]]["correct_doc_id"] for v in l]
    agg = diagnosticity(allc, allw)
    print(f"pooled: P(solve|correct)={agg['p_correct']:.3f}  P(solve|wrong)={agg['p_wrong']:.3f}  "
          f"Delta={agg['delta']:.3f}   (n={len(allc)} / {len(allw)})")

    groups = defaultdict(list)
    for tid, d in deltas.items():
        groups[bucket_of(d["delta"])].append(tid)
    order = ["none (<=0.05)", "low (0.05-0.30)", "medium (0.30-0.60)", "high (>=0.60)"]

    print(f"\n{'bucket':<22}{'tasks':>7}{'mean Delta':>12}{'share':>8}")
    print("-" * 100)
    tot = sum(len(v) for v in groups.values())
    for b in order:
        ids = groups.get(b, [])
        if not ids:
            continue
        md = st.mean([deltas[i]["delta"] for i in ids])
        print(f"{b:<22}{len(ids):>7}{md:>12.3f}{len(ids)/tot*100:>7.1f}%")

    print("\n" + "=" * 100)
    print(f"RRL GAIN BY BUCKET   arm='{args.arm}'  vs '{args.baseline}'  epochs={args.epochs}")
    print("Feedback is the real verifier on held-out generations.")
    print("=" * 100)
    print(f"{'bucket':<22}{'tasks':>6}{'Delta':>7}{'base':>8}{'RRL':>8}{'gain':>8}{'p':>9}{'cov':>7}{'solve base':>12}{'solve RRL':>11}")
    print("-" * 100)
    for b in order:
        ids = set(groups.get(b, []))
        if len(ids) < 5:
            if ids:
                print(f"{b:<22}{len(ids):>6}   (too few tasks to score)")
            continue
        base_h, rrl_h, base_s, rrl_s, cov = [], [], [], [], []
        for s in seeds:
            rb = run(s, pre, ARMS[args.baseline], ids, ev, args.epochs)
            rr = run(s, pre, ARMS[args.arm], ids, ev, args.epochs)
            if not rb or not rr:
                continue
            base_h.append(st.mean(rb["hit"]))
            rrl_h.append(st.mean(rr["hit"]))
            if rb["solved"]:
                base_s.append(st.mean(rb["solved"]))
            if rr["solved"]:
                rrl_s.append(st.mean(rr["solved"]))
            cov.append(rr["coverage"])
        if not base_h:
            continue
        d, p = paired(rrl_h, base_h)
        md = st.mean([deltas[i]["delta"] for i in ids])
        print(f"{b:<22}{len(ids):>6}{md:>7.2f}{st.mean(base_h)*100:>8.1f}{st.mean(rrl_h)*100:>8.1f}"
              f"{d*100:>+8.1f}{p:>9.4f}{st.mean(cov)*100:>6.0f}%"
              f"{(st.mean(base_s)*100 if base_s else float('nan')):>12.1f}"
              f"{(st.mean(rrl_s)*100 if rrl_s else float('nan')):>11.1f}")
    print("-" * 100)
    print("'solve' columns are downstream task success, the metric the old benchmark used.")
    print("=" * 100)


if __name__ == "__main__":
    main()
