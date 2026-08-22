"""
RRL benchmark: every arm sees the same candidates, scored on ranking metrics.

Design rules this harness enforces, because breaking any of them silently invalidates the
comparison:

  * Identical candidate sets. Every arm reranks the same top-K shortlist drawn by the same
    base retriever. Comparing a corpus-wide scorer against a shortlist reranker measures
    candidate-set size, not ranking quality.
  * Ranking metrics first. Hit@1 / MRR / nDCG@5 are observed once per query, so a sweep
    yields thousands of observations. Task pass rate is reported too, but it is a weak
    downstream proxy: the same retrieval improvement shows up in it scaled by the
    verifier's diagnosticity.
  * Provenance-checked outcomes. Pass rates come only from a strict, real-generation
    cache; a cache miss is reported as reduced coverage, never imputed.
  * Verifier diagnosticity reported alongside. Delta decides whether outcome feedback can
    identify a document at all, so it belongs next to the result, not in an appendix.

Usage:
    python3 sim/run_bench.py --seeds 10 --epochs 16 --verifier oracle
    python3 sim/run_bench.py --seeds 10 --epochs 16 --verifier cache --tasks sensitive
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
from typing import Dict, List, Optional, Tuple

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from rrl.store import Candidate, CandidateStore
from rrl.layer import ReputationLayer
from rrl.retriever import Retriever, BM25, tokenize
from rrl.ingest import Ingester
from rrl.feedback import (
    Attribution,
    OutcomeSignals,
    VerifierNoise,
    calculate_outcome,
    update_counters,
)
from rrl.metrics import hit_at_k, reciprocal_rank, ndcg_at_k, diagnosticity
from run_gate_recurring import build_dataset, topic
from outcome_cache import OutcomeCache

DATA = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
SHORTLIST = 5

# ---------------------------------------------------------------- outcomes


def load_outcomes(graded: bool) -> Dict[Tuple[int, str], float]:
    """Provenance-checked outcome table. Graded uses the per-assert fraction."""
    path = os.path.join(DATA, "gemini_cache_graded.jsonl" if graded else "gemini_cache.jsonl")
    cache = OutcomeCache(path).load()
    return {k: (r.fraction if graded else float(r.passed)) for k, r in cache.records.items()}


def sensitive_tasks(outcomes: Dict[Tuple[int, str], float]) -> set:
    """
    Tasks whose outcome actually depends on which document was retrieved.

    A task the model solves regardless of evidence, or fails regardless, contributes no
    signal a reputation layer could learn from — and worse, it sets the sample-complexity
    floor for every other task by depressing Delta.
    """
    own, oth = defaultdict(list), defaultdict(list)
    for (tid, did), v in outcomes.items():
        (own if did.startswith(f"good_{tid}_") else oth)[tid].append(v)
    keep = set()
    for tid in own:
        if not oth.get(tid):
            continue
        if st.mean(own[tid]) > st.mean(oth[tid]):
            keep.add(tid)
    return keep


# ---------------------------------------------------------------- precompute


def precompute(seeds: List[int], cache_path: str):
    if os.path.exists(cache_path):
        return pickle.load(open(cache_path, "rb"))
    from sentence_transformers import SentenceTransformer, CrossEncoder

    model = SentenceTransformer("all-MiniLM-L6-v2")
    cross = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    out = {}
    for seed in seeds:
        problems, docs = build_dataset(seed)
        store = CandidateStore()
        ing = Ingester(model=model)
        for did, (code, _) in docs.items():
            ing.ingest_document(store, did, code)
        cands = store.list_candidates()
        retr = Retriever(store, model=model)
        bm = BM25(cands)
        contents = {c.id: c.content for c in cands}
        per = {}
        embs = model.encode([p["text"] for p in problems])
        for p, qe in zip(problems, embs):
            qe = qe.tolist()
            vec = {}
            for c in cands:
                e = c.metadata["embedding"]
                dot = sum(a * b for a, b in zip(qe, e))
                nq = sum(a * a for a in qe) ** 0.5
                nc = sum(b * b for b in e) ** 0.5
                vec[c.id] = dot / (nq * nc) if nq * nc > 0 else 0.0
            dense = retr._normalize_scores(vec)
            fused = retr._normalize_scores(
                retr._compute_rrf(vec, bm.get_scores(tokenize(p["text"])))
            )
            short = sorted(fused, key=lambda k: -fused[k])[:SHORTLIST]
            raw = cross.predict([(p["text"], contents[k]) for k in short])
            lo, hi = float(min(raw)), float(max(raw))
            rng = (hi - lo) or 1.0
            ce = {k: float((v - lo) / rng) for k, v in zip(short, raw)}
            per[p["task_id"]] = {
                "dense": {k: dense[k] for k in short},
                "fused": {k: fused[k] for k in short},
                "ce": ce,
                "short": short,
                "text": p["text"],
            }
        out[seed] = {"problems": problems, "contents": contents, "per": per}
    pickle.dump(out, open(cache_path, "wb"))
    return out


# ---------------------------------------------------------------- arms

# Each arm: base score source, reputation config. None config = static, no learning.
ARMS = {
    "B0 dense only": dict(base="dense", rep=None),
    "B1 cross-encoder": dict(base="ce", rep=None),
    "B2 RRF hybrid": dict(base="fused", rep=None),
    "B3 RRF + static prior": dict(base="fused", rep=None, static_prior=True),
    "B4 RRF + Thompson": dict(base="fused", rep=dict(w=(0.70, 0.30, 0.00, 0.10))),
    "B5 CE + Thompson": dict(base="ce", rep=dict(w=(0.70, 0.30, 0.00, 0.10))),
    "B6 CE + RRL": dict(base="ce", rep=dict(w=(0.70, 0.20, 0.10, 0.10))),
    "B7 CE + RRL + pooling": dict(
        base="ce", rep=dict(w=(0.70, 0.20, 0.10, 0.10), cond="query")
    ),
    "B8 RRF + RRL + pooling": dict(
        base="fused", rep=dict(w=(0.70, 0.20, 0.10, 0.10), cond="query")
    ),
    "B9 CE + RRL + noise corr.": dict(
        base="ce", rep=dict(w=(0.70, 0.20, 0.10, 0.10), cond="query", noise=True)
    ),
    # Dense similarity turns out to be the strongest base ranker on this corpus, so the
    # layer has to prove itself on top of that, not on top of a weaker fused score.
    "B10 dense + RRL": dict(base="dense", rep=dict(w=(0.70, 0.20, 0.10, 0.10))),
    "B11 dense + RRL + pooling": dict(
        base="dense", rep=dict(w=(0.70, 0.20, 0.10, 0.10), cond="query")
    ),
    "B12 dense + RRL + topic pool": dict(
        base="dense", rep=dict(w=(0.70, 0.20, 0.10, 0.10), cond="topic")
    ),
}


def query_stream(seed: int, problems: List[dict], epochs: int) -> List[dict]:
    rng = random.Random(seed)
    out = []
    for _ in range(epochs):
        order = list(problems)
        rng.shuffle(order)
        out.extend(order)
    return out


def run_arm(
    seed: int,
    pre: dict,
    arm: dict,
    *,
    epochs: int,
    outcomes: Dict[Tuple[int, str], float],
    verifier: str,
    noise: Optional[VerifierNoise],
    task_filter: Optional[set],
    warmup: float,
) -> dict:
    problems = pre["problems"]
    if task_filter is not None:
        problems = [p for p in problems if p["task_id"] in task_filter]
    if not problems:
        return {}
    contents, per = pre["contents"], pre["per"]
    random.seed(seed)

    rep = arm.get("rep")
    layer = None
    if rep is not None:
        store = CandidateStore()
        for cid in contents:
            store.add_candidate(Candidate(id=cid, content=contents[cid], last_confirmed=0.0))
        layer = ReputationLayer(
            store,
            weights=rep["w"],
            gamma=1.0,
            decay_unit_sec=1.0,
            exploration_mode="ts",
            warmup_observations=warmup,
        )

    static_prior = None
    if arm.get("static_prior"):
        # Frozen, query-independent document quality. Isolates "having a prior" from
        # "updating one online": if B3 matches the adaptive arms, adaptivity is not
        # what is doing the work.
        acc = defaultdict(list)
        for (tid, did), v in outcomes.items():
            acc[did].append(v)
        static_prior = {d: st.mean(v) for d, v in acc.items()}

    hits, rrs, ndcgs, passes, misses, selections = [], [], [], [], 0, []
    for step, p in enumerate(query_stream(seed, problems, epochs)):
        tid = p["task_id"]
        info = per[tid]
        src = dict(info[arm["base"]])
        relevance = {d: 1.0 for d in info["short"] if d.startswith(f"good_{tid}_")}

        if layer is None:
            scored = dict(src)
            if static_prior:
                scored = {
                    d: 0.70 * s + 0.30 * static_prior.get(d, 0.5) for d, s in src.items()
                }
            ranking = sorted(scored, key=lambda k: -scored[k])
            cluster = None
        else:
            cluster = f"q{tid}" if rep.get("cond") == "query" else (
                topic(info["text"]) if rep.get("cond") == "topic" else None
            )
            res = layer.rescore(
                src,
                top_k=len(src),
                explore=True,
                now=float(step),
                cluster_id=cluster,
                epsilon=0.0,
                gamma=1.0,
                decay_unit_sec=1.0,
            )
            ranking = [cid for cid, _, _ in res.results]

        pick = ranking[0]
        selections.append(pick)
        hits.append(hit_at_k(ranking, list(relevance), 1))
        rrs.append(reciprocal_rank(ranking, list(relevance)))
        ndcgs.append(ndcg_at_k(ranking, relevance, 5))

        observed = outcomes.get((tid, pick))
        if observed is None:
            misses += 1
        else:
            passes.append(observed)

        if layer is not None:
            if verifier == "oracle":
                y = hits[-1]
            else:
                if observed is None:
                    continue  # unknown outcome is not evidence; never impute it
                y = observed
            sig = OutcomeSignals(
                s_gt=y,
                attribution=Attribution.RETRIEVAL,
                noise=noise if rep.get("noise") else None,
            )
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
                cluster_id=cluster,
            )

    n_q = len(problems)
    return {
        "hit": hits,
        "mrr": rrs,
        "ndcg": ndcgs,
        "pass": passes,
        "coverage": 1.0 - misses / max(1, len(hits)),
        "n_queries": n_q,
        "selections": selections,
    }


# ---------------------------------------------------------------- stats


def ci95(vals: List[float]) -> Tuple[float, float, float]:
    if not vals:
        return 0.0, 0.0, 0.0
    m = st.mean(vals)
    if len(vals) < 2:
        return m, m, m
    sem = st.stdev(vals) / math.sqrt(len(vals))
    try:
        from scipy import stats as sp

        t = float(sp.t.ppf(0.975, len(vals) - 1))
    except Exception:
        t = 2.262
    return m, m - t * sem, m + t * sem


def paired(a: List[float], b: List[float]) -> Tuple[float, float]:
    d = [x - y for x, y in zip(a, b)]
    if len(d) < 2:
        return 0.0, float("nan")
    m = st.mean(d)
    sem = st.stdev(d) / math.sqrt(len(d))
    if sem == 0:
        return m, 0.0 if m != 0 else 1.0
    t = m / sem
    try:
        from scipy import stats as sp

        return m, float(2 * (1 - sp.t.cdf(abs(t), len(d) - 1)))
    except Exception:
        return m, float("nan")


# ---------------------------------------------------------------- main


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=16)
    ap.add_argument(
        "--verifier",
        choices=["oracle", "cache"],
        default="oracle",
        help="oracle = perfect feedback (mechanism ceiling); cache = real unit-test outcomes",
    )
    ap.add_argument("--graded", action="store_true", help="use per-assert fractions")
    ap.add_argument(
        "--tasks",
        choices=["all", "sensitive"],
        default="all",
        help="sensitive = drop tasks whose outcome does not depend on the retrieved document",
    )
    ap.add_argument("--warmup", type=float, default=0.0)
    ap.add_argument("--baseline", default="B1 cross-encoder")
    args = ap.parse_args()

    seeds = list(range(42, 42 + args.seeds))
    pre = precompute(seeds, os.path.join("/private/tmp", f"rrl_bench_{args.seeds}.pkl"))
    outcomes = load_outcomes(args.graded)

    tfilter = sensitive_tasks(outcomes) if args.tasks == "sensitive" else None

    # Diagnosticity of the verifier actually in use, on the task set actually in use.
    own, oth = [], []
    for (tid, did), v in outcomes.items():
        if tfilter is not None and tid not in tfilter:
            continue
        (own if did.startswith(f"good_{tid}_") else oth).append(v)
    diag = diagnosticity(own, oth)

    print("=" * 104)
    print(
        f"RRL BENCHMARK  seeds={args.seeds} epochs={args.epochs} shortlist={SHORTLIST} "
        f"verifier={args.verifier}{' graded' if args.graded else ''} tasks={args.tasks}"
    )
    print(
        f"verifier diagnosticity on this task set: P(pass|correct)={diag['p_correct']:.3f} "
        f"P(pass|wrong)={diag['p_wrong']:.3f} Delta={diag['delta']:.3f} "
        f"-> {diag['n_required']:.0f} obs/doc needed ({diag['n_required']*SHORTLIST:.0f} epochs)"
    )
    print(
        f"observations actually available: {args.epochs/SHORTLIST:.1f} obs/doc"
        f"   [shortfall {diag['n_required']/(args.epochs/SHORTLIST):.0f}x]"
        if diag["n_required"] != float("inf")
        else ""
    )
    print("=" * 104)
    hdr = f"{'arm':<28} {'Hit@1':>16} {'MRR':>8} {'nDCG@5':>8} {'pass':>8} {'cov':>6} {'ep1->epN':>12}"
    print(hdr)
    print("-" * 104)

    noise = VerifierNoise(rho_fp=max(0.0, diag["p_wrong"]), rho_fn=max(0.0, 1.0 - diag["p_correct"]))
    results = {}
    for name, arm in ARMS.items():
        per_seed_hit, per_seed_mrr, per_seed_ndcg, per_seed_pass, cov = [], [], [], [], []
        curves = []
        for s in seeds:
            r = run_arm(
                s,
                pre[s],
                arm,
                epochs=args.epochs,
                outcomes=outcomes,
                verifier=args.verifier,
                noise=noise,
                task_filter=tfilter,
                warmup=args.warmup,
            )
            if not r:
                continue
            per_seed_hit.append(st.mean(r["hit"]))
            per_seed_mrr.append(st.mean(r["mrr"]))
            per_seed_ndcg.append(st.mean(r["ndcg"]))
            if r["pass"]:
                per_seed_pass.append(st.mean(r["pass"]))
            cov.append(r["coverage"])
            curves.append((r["hit"], r["n_queries"]))
        if not per_seed_hit:
            continue
        results[name] = dict(hit=per_seed_hit, mrr=per_seed_mrr, ndcg=per_seed_ndcg)
        m, lo, hi = ci95(per_seed_hit)
        nq = curves[0][1]
        ep1 = st.mean([st.mean(c[:nq]) for c, _ in curves])
        epN = st.mean([st.mean(c[-nq:]) for c, _ in curves])
        pv = st.mean(per_seed_pass) if per_seed_pass else float("nan")
        print(
            f"{name:<28} {m*100:5.1f} [{lo*100:4.1f},{hi*100:4.1f}] {st.mean(per_seed_mrr):8.3f} "
            f"{st.mean(per_seed_ndcg):8.3f} {pv:8.3f} {st.mean(cov)*100:5.0f}% "
            f"{ep1*100:5.1f}->{epN*100:5.1f}"
        )

    base = args.baseline
    if base in results:
        print("-" * 104)
        print(f"paired against {base} across the same {len(seeds)} seeds (Hit@1):")
        for name, r in results.items():
            if name == base:
                continue
            d, p = paired(r["hit"], results[base]["hit"])
            flag = "*" if p < 0.05 else " "
            print(f"  {flag} {name:<28} {d*100:+6.1f} pts   p={p:.4f}")
    print("=" * 104)


if __name__ == "__main__":
    main()
