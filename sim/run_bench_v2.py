"""
Phase 3: RRL on the frozen, externally annotated benchmark.

Every arm ranks the *same* fixed 5 candidates, so nothing here can be explained by one arm
seeing a different candidate set. Ground truth is CodeRAG-Bench's manual annotation, not
ours. The verifier is an oracle by default, which needs no generation at all -- that isolates
"can the mechanism learn" from "is the feedback channel informative", which the MBPP results
showed are entirely different questions.

Usage:
  python3 sim/run_bench_v2.py --seeds 8 --epochs 16
  python3 sim/run_bench_v2.py --seeds 8 --epochs 64 --curve
  python3 sim/run_bench_v2.py --seeds 8 --epochs 32 --tiers
"""

import argparse
import json
import math
import os
import pickle
import random
import statistics as st
import sys
from collections import defaultdict

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from rrl.store import Candidate, CandidateStore
from rrl.layer import ReputationLayer
from rrl.retriever import Retriever, BM25, tokenize
from rrl.feedback import Attribution, OutcomeSignals, calculate_outcome, update_counters
from rrl.metrics import hit_at_k, reciprocal_rank, ndcg_at_k

BENCH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "benchmark"))
PRE = "/private/tmp/rrl_bench_v2_pre.pkl"


def load_bench():
    tasks = [json.loads(l) for l in open(os.path.join(BENCH, "rrl_tasks.jsonl")) if l.strip()]
    docs = {
        d["doc_id"]: d
        for d in (
            json.loads(l) for l in open(os.path.join(BENCH, "rrl_candidates.jsonl")) if l.strip()
        )
    }
    return tasks, docs


def precompute():
    if os.path.exists(PRE):
        return pickle.load(open(PRE, "rb"))
    from sentence_transformers import SentenceTransformer, CrossEncoder

    tasks, docs = load_bench()
    model = SentenceTransformer("all-MiniLM-L6-v2")
    cross = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    needed = sorted({c for t in tasks for c in t["candidate_doc_ids"]})
    d_emb = dict(zip(needed, model.encode([docs[c]["text"] for c in needed]).tolist()))
    q_emb = dict(
        zip(
            [t["task_id"] for t in tasks],
            model.encode([t["query"] for t in tasks]).tolist(),
        )
    )
    bm_docs = [Candidate(id=c, content=docs[c]["text"]) for c in needed]
    bm = BM25(bm_docs)
    dummy = Retriever(CandidateStore(), model=model)

    per = {}
    for t in tasks:
        cands = t["candidate_doc_ids"]
        qe = q_emb[t["task_id"]]
        nq = sum(a * a for a in qe) ** 0.5
        vec = {}
        for c in cands:
            e = d_emb[c]
            nc = sum(b * b for b in e) ** 0.5
            vec[c] = (sum(a * b for a, b in zip(qe, e)) / (nq * nc)) if nq * nc else 0.0
        bm_all = bm.get_scores(tokenize(t["query"]))
        bm_sub = {c: bm_all.get(c, 0.0) for c in cands}
        raw = cross.predict([(t["query"], docs[c]["text"]) for c in cands])
        lo, hi = float(min(raw)), float(max(raw))
        rng = (hi - lo) or 1.0
        per[t["task_id"]] = {
            "dense": dummy._normalize_scores(vec),
            "bm25": dummy._normalize_scores(bm_sub),
            "fused": dummy._normalize_scores(dummy._compute_rrf(vec, bm_sub)),
            "ce": {c: float((v - lo) / rng) for c, v in zip(cands, raw)},
        }
    out = {"tasks": tasks, "per": per, "doc_ids": needed}
    pickle.dump(out, open(PRE, "wb"))
    return out


ARMS = {
    "dense only": dict(base="dense", rep=None),
    "BM25 only": dict(base="bm25", rep=None),
    "RRF hybrid": dict(base="fused", rep=None),
    "cross-encoder": dict(base="ce", rep=None),
    "dense + Thompson": dict(base="dense", rep=dict(w=(0.70, 0.30, 0.00, 0.10))),
    "dense + RRL": dict(base="dense", rep=dict(w=(0.70, 0.20, 0.10, 0.10))),
    "dense + RRL + pooling": dict(base="dense", rep=dict(w=(0.70, 0.20, 0.10, 0.10), cond=True)),
    "CE + RRL + pooling": dict(base="ce", rep=dict(w=(0.70, 0.20, 0.10, 0.10), cond=True)),
    "RRF + RRL + pooling": dict(base="fused", rep=dict(w=(0.70, 0.20, 0.10, 0.10), cond=True)),
}


def run_arm(seed, pre, arm, epochs, warmup=0.0, delta=None):
    tasks, per = pre["tasks"], pre["per"]
    random.seed(seed)
    rng = random.Random(seed * 7919)
    rep = arm.get("rep")
    layer = None
    if rep is not None:
        store = CandidateStore()
        for c in pre["doc_ids"]:
            store.add_candidate(Candidate(id=c, content="", last_confirmed=0.0))
        layer = ReputationLayer(
            store,
            weights=rep["w"],
            gamma=1.0,
            decay_unit_sec=1.0,
            exploration_mode="ts",
            warmup_observations=warmup,
        )
    order_rng = random.Random(seed)
    stream = []
    for _ in range(epochs):
        o = list(tasks)
        order_rng.shuffle(o)
        stream.extend(o)

    hits, rrs, nds = [], [], []
    tier_pick = defaultdict(int)
    for step, t in enumerate(stream):
        src = dict(per[t["task_id"]][arm["base"]])
        correct = t["correct_doc_id"]
        relevance = {correct: 1.0}
        if layer is None:
            ranking = sorted(src, key=lambda k: -src[k])
        else:
            cl = t["task_id"] if rep.get("cond") else None
            ranking = [
                cid
                for cid, _, _ in layer.rescore(
                    src,
                    top_k=len(src),
                    explore=True,
                    now=float(step),
                    cluster_id=cl,
                    epsilon=0.0,
                    gamma=1.0,
                    decay_unit_sec=1.0,
                ).results
            ]
        pick = ranking[0]
        ok = pick == correct
        hits.append(1.0 if ok else 0.0)
        rrs.append(reciprocal_rank(ranking, [correct]))
        nds.append(ndcg_at_k(ranking, relevance, 5))
        if not ok:
            tiers = t["distractor_tiers"]
            for name, ids in (("tier1", tiers["tier1_same_family"]),
                              ("tier2", tiers["tier2_other_family"]),
                              ("tier3", tiers["tier3_irrelevant"])):
                if pick in ids:
                    tier_pick[name] += 1
        if layer is not None:
            if delta is None:
                y = 1.0 if ok else 0.0
            else:
                p1, p2 = 0.5 + delta / 2.0, 0.5 - delta / 2.0
                y = 1.0 if rng.random() < (p1 if ok else p2) else 0.0
            sig = OutcomeSignals(s_gt=y, attribution=Attribution.RETRIEVAL)
            update_counters(
                layer.store,
                {pick: src[pick]},
                calculate_outcome(sig, use_safeguards=True),
                current_timestamp=float(step),
                gamma=1.0,
                decay_unit_sec=1.0,
                credit_smoothing=0.50,
                use_liar_counter=True,
                signals=sig,
                cluster_id=(t["task_id"] if rep.get("cond") else None),
            )
    return {"hit": hits, "mrr": rrs, "ndcg": nds, "tiers": dict(tier_pick), "n_q": len(tasks)}


def ci(v):
    if len(v) < 2:
        return (v[0] if v else 0.0, 0.0, 0.0)
    m = st.mean(v)
    sem = st.stdev(v) / math.sqrt(len(v))
    try:
        from scipy import stats as sp

        tt = float(sp.t.ppf(0.975, len(v) - 1))
    except Exception:
        tt = 2.365
    return m, m - tt * sem, m + tt * sem


def paired(a, b):
    d = [x - y for x, y in zip(a, b)]
    m = st.mean(d)
    sem = st.stdev(d) / math.sqrt(len(d)) if len(d) > 1 else 0.0
    if sem == 0:
        return m, float("nan")
    try:
        from scipy import stats as sp

        return m, float(2 * (1 - sp.t.cdf(abs(m / sem), len(d) - 1)))
    except Exception:
        return m, float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=16)
    ap.add_argument("--baseline", default="dense only")
    ap.add_argument("--curve", action="store_true")
    ap.add_argument("--tiers", action="store_true")
    ap.add_argument("--warmup", type=float, default=0.0)
    ap.add_argument("--delta-sweep", type=str, default="",
                    help="comma-separated deltas; sweeps verifier diagnosticity on this benchmark")
    ap.add_argument("--sweep-arm", default="dense + RRL + pooling")
    args = ap.parse_args()

    pre = precompute()
    man = json.load(open(os.path.join(BENCH, "manifest.json")))
    seeds = list(range(42, 42 + args.seeds))

    if args.delta_sweep:
        nq = len(pre["tasks"])
        static = st.mean(
            [st.mean(run_arm(s, pre, ARMS["dense only"], args.epochs)["hit"]) for s in seeds]
        )
        arm = ARMS[args.sweep_arm]
        print("=" * 100)
        print(
            f"DELTA SWEEP on the frozen benchmark  arm='{args.sweep_arm}'  "
            f"tasks={nq} seeds={args.seeds} epochs={args.epochs}"
        )
        print(f"static dense baseline Hit@1 = {static*100:.1f}%")
        print("=" * 100)
        print(f"{'Delta':>6} {'final Hit@1':>12} {'gain':>8} {'ep1':>7} {'ep4':>7} {'ep8':>7} {'ep16':>7} {'epN':>7}")
        print("-" * 100)
        for d in [float(x) for x in args.delta_sweep.split(",")]:
            cs = [run_arm(s, pre, arm, args.epochs, delta=d)["hit"] for s in seeds]
            ep = lambda e: st.mean([st.mean(c[(e - 1) * nq : e * nq]) for c in cs])
            fin = st.mean([st.mean(c[-nq:]) for c in cs])
            marks = [ep(e) for e in (1, 4, 8, 16) if e <= args.epochs]
            marks += [0.0] * (4 - len(marks))
            print(
                f"{d:>6.2f} {fin*100:11.1f}% {(fin-static)*100:+7.1f} "
                + " ".join(f"{m*100:6.1f}" for m in marks)
                + f" {fin*100:6.1f}"
            )
        print("=" * 100)
        return

    print("=" * 100)
    print(
        f"RRL BENCHMARK v2 (frozen, externally annotated)   tasks={man['counts']['tasks']} "
        f"candidates/task={man['construction']['candidates_per_task']} "
        f"seeds={args.seeds} epochs={args.epochs}"
    )
    print(f"ground truth: {man['ground_truth_source']['dataset']} ({man['ground_truth_source']['paper']})")
    print(f"tasks sha256={man['files']['rrl_tasks.jsonl'][:16]}  verifier=oracle")
    print("=" * 100)
    print(f"{'arm':<26} {'Hit@1':>18} {'MRR':>8} {'nDCG@5':>8} {'ep1':>7} {'epN':>7}")
    print("-" * 100)

    res, curves = {}, {}
    for name, arm in ARMS.items():
        h, m, n, cs = [], [], [], []
        tier_tot = defaultdict(int)
        for s in seeds:
            r = run_arm(s, pre, arm, args.epochs, warmup=args.warmup)
            h.append(st.mean(r["hit"]))
            m.append(st.mean(r["mrr"]))
            n.append(st.mean(r["ndcg"]))
            cs.append(r["hit"])
            for k, v in r["tiers"].items():
                tier_tot[k] += v
        res[name] = h
        curves[name] = (cs, tier_tot)
        nq = pre["per"] and len(pre["tasks"])
        mm, lo, hi = ci(h)
        ep1 = st.mean([st.mean(c[:nq]) for c in cs])
        epN = st.mean([st.mean(c[-nq:]) for c in cs])
        print(
            f"{name:<26} {mm*100:5.1f} [{lo*100:4.1f},{hi*100:4.1f}] {st.mean(m):8.3f} "
            f"{st.mean(n):8.3f} {ep1*100:6.1f} {epN*100:6.1f}"
        )

    base = args.baseline
    print("-" * 100)
    print(f"paired vs '{base}' across the same {len(seeds)} seeds (Hit@1):")
    for name, h in res.items():
        if name == base:
            continue
        d, p = paired(h, res[base])
        print(f"  {'*' if p < 0.05 else ' '} {name:<26} {d*100:+6.1f} pts   p={p:.4f}")

    if args.tiers:
        print("\n" + "=" * 100)
        print("WHICH DISTRACTOR WINS WHEN AN ARM IS WRONG (share of errors)")
        print("=" * 100)
        print(f"{'arm':<26} {'tier1 same-family':>19} {'tier2 related':>15} {'tier3 irrelevant':>18}")
        print("-" * 100)
        for name, (_, tt) in curves.items():
            tot = sum(tt.values()) or 1
            print(
                f"{name:<26} {tt.get('tier1',0)/tot*100:18.1f}% "
                f"{tt.get('tier2',0)/tot*100:14.1f}% {tt.get('tier3',0)/tot*100:17.1f}%"
            )

    if args.curve:
        nq = len(pre["tasks"])
        print("\n" + "=" * 100)
        print("RECURRENCE CURVE (Hit@1 by epoch)")
        print("=" * 100)
        marks = [e for e in (1, 2, 4, 8, 16, 32, 64, 128) if e <= args.epochs]
        print(f"{'arm':<26}" + "".join(f"{('ep'+str(e)):>8}" for e in marks))
        print("-" * 100)
        for name, (cs, _) in curves.items():
            row = ""
            for e in marks:
                v = st.mean([st.mean(c[(e - 1) * nq : e * nq]) for c in cs])
                row += f"{v*100:8.1f}"
            print(f"{name:<26}{row}")
    print("=" * 100)


if __name__ == "__main__":
    main()
