"""
Gate T: tool selection for agents, on real tool descriptions and real requests (ToolRet).

An agent must pick one tool for each request from a candidate pool. RRL learns which tools
actually work, from outcomes, and shares that across requests. The data is ToolRet
(Shi et al., Findings of ACL 2025, huggingface.co/datasets/mangopy/ToolRet-*): the ToolBench
and ToolLens request sets, whose gold labels say which web-API tools solve each request, and
whose tools are reused across requests.

On top of the real data the benchmark adds the failure modes of real tool ecosystems:
  * Tool updates (as in MCPEvol-Bench, arXiv 2607.14642): 30% of the gold tools get a v2
    replacement. Before the switch only v1 works; after it, only v2 does.
  * Poisoned tools (as in MCPTox, AAAI 2026): 10% of the gold tools get a malicious twin whose
    description copies the wording of requests the real tool serves. Choosing it always fails.
  * Flaky tools: 20% of tools fail half their calls (timeouts, rate limits); the rest fail 5%.
  * Held-out requests: 20% of requests appear only once, in a final block.

Success = the chosen tool is a gold tool for the request (and is deployed and does not flake).
This is ToolRet's label, standing in for real execution: the APIs are not called. That is the
main limitation of this gate.

ToolRet's dataset card states no license, so the raw data is downloaded at build time into
data/gate_t/raw/ (git-ignored) and only derived ids, scores and labels are committed.

    python3 sim/run_gate_t.py --build        # download ToolRet, build pools (needs network)
    python3 sim/run_gate_t.py --scenario all # replay offline from data/gate_t/
    python3 sim/run_gate_t.py --selftest
"""

import argparse
import json
import math
import os
import random
import re
import sys
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from run_gate_matrix import (
    NO_DOC,
    RESULTS_DIR,
    enable_ablation,
    plot,
    print_report,
    run_policies,
    set_semantic,
    summarize,
)

HERE = os.path.dirname(os.path.abspath(__file__))
T_DIR = os.path.abspath(os.path.join(HERE, "..", "data", "gate_t"))
RAW_DIR = os.path.join(T_DIR, "raw")

SUBSETS = {"toolbench": "toolbench_tool_", "toollens": "toolLens_tool_"}
MAX_QUERIES = {"toolbench": 600, "toollens": 314}
POOL_K = 8
UPDATE_FRAC, POISON_FRAC, FLAKY_FRAC = 0.3, 0.1, 0.2
FLAKY_FAIL, BASE_FAIL = 0.5, 0.05
HELDOUT_FRAC = 0.2

# Exact ToolRet revisions the committed pools were built from (pinned for reproducibility, and so
# that a changed upstream repository cannot silently change the benchmark).
TOOLRET_REVISIONS = {
    "mangopy/ToolRet-Queries": "b8c76ad3349ff17497b6bdb28bb5b8f61a0f6445",
    "mangopy/ToolRet-Tools": "e06c38c75612b6536bd959e08cdd345894aba6a7",
}

SCENARIOS: Dict[str, dict] = {
    "stationary": {"coverage": 1.0, "sycophancy": 0.0, "judge_acc": 0.8, "switch": False},
    "sparse_verifier": {"coverage": 0.2, "sycophancy": 0.0, "judge_acc": 0.8, "switch": False},
    "sycophantic": {"coverage": 0.2, "sycophancy": 0.5, "judge_acc": 0.8, "switch": False},
    # Poisoned tools return plausible-looking output, so the agent's own check accepts it 90%
    # of the time.
    "fooled": {
        "coverage": 0.2,
        "sycophancy": 0.0,
        "judge_acc": 0.8,
        "switch": False,
        "fooled_prob": 0.9,
    },
    "drift": {"coverage": 1.0, "sycophancy": 0.0, "judge_acc": 0.8, "switch": True},
    "drift_sycophantic": {"coverage": 0.2, "sycophancy": 0.5, "judge_acc": 0.8, "switch": True},
    "drift_fooled": {
        "coverage": 0.2,
        "sycophancy": 0.0,
        "judge_acc": 0.8,
        "switch": True,
        "fooled_prob": 0.9,
    },
}


# ---------------------------------------------------------------- build (network, untrusted)


def download() -> None:
    from huggingface_hub import hf_hub_download

    files = [("mangopy/ToolRet-Tools", "web/tools-00000-of-00001.parquet")] + [
        ("mangopy/ToolRet-Queries", f"{s}/queries-00000-of-00001.parquet") for s in SUBSETS
    ]
    for repo, fn in files:
        hf_hub_download(
            repo,
            fn,
            repo_type="dataset",
            revision=TOOLRET_REVISIONS[repo],
            local_dir=os.path.join(RAW_DIR, repo),
        )


def _tool_text(doc_json: str) -> Tuple[str, str]:
    """(name, searchable text) from a ToolRet documentation field."""
    try:
        doc = json.loads(doc_json)
    except Exception:
        return "", doc_json[:1000]
    name = str(doc.get("name", ""))
    parts = [name, str(doc.get("category_name", "")), str(doc.get("description", ""))]
    for key in ("required_parameters", "optional_parameters"):
        for p in doc.get(key, []) or []:
            if isinstance(p, dict):
                parts.append(f"{p.get('name', '')}: {p.get('description', '')}")
    return name, " ".join(x for x in parts if x).strip()[:1000]


def _bm25(corpus_tokens: List[List[str]], k1: float = 1.5, b: float = 0.75):
    n = len(corpus_tokens)
    avg = sum(len(t) for t in corpus_tokens) / max(1, n)
    df: Counter = Counter()
    for toks in corpus_tokens:
        df.update(set(toks))
    idf = {w: math.log(1 + (n - c + 0.5) / (c + 0.5)) for w, c in df.items()}
    tfs = [Counter(t) for t in corpus_tokens]
    lens = [len(t) for t in corpus_tokens]

    def scores(q: List[str]) -> List[float]:
        out = []
        for tf, ln in zip(tfs, lens):
            s = 0.0
            for w in q:
                f = tf.get(w, 0)
                if f:
                    s += idf[w] * f * (k1 + 1) / (f + k1 * (1 - b + b * ln / avg))
            out.append(s)
        return out

    return scores


def _tok(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def build_subset(subset: str, embedder, cross_encoder) -> dict:
    import numpy as np
    import pandas as pd

    rng = random.Random(2026)
    tools = pd.read_parquet(
        os.path.join(RAW_DIR, "mangopy/ToolRet-Tools/web/tools-00000-of-00001.parquet")
    )
    tools = tools[tools["id"].str.startswith(SUBSETS[subset])]
    text = {}
    name = {}
    for tid, doc in zip(tools["id"], tools["documentation"]):
        name[tid], text[tid] = _tool_text(doc)
    queries = pd.read_parquet(
        os.path.join(RAW_DIR, f"mangopy/ToolRet-Queries/{subset}/queries-00000-of-00001.parquet")
    )
    qrows = []
    for qid, qtext, labels in zip(queries["id"], queries["query"], queries["labels"]):
        gold = [
            x["id"] for x in json.loads(labels) if x.get("relevance", 1) > 0 and x["id"] in text
        ]
        if gold:
            qrows.append({"qid": qid, "text": qtext, "gold": gold})
    rng.shuffle(qrows)
    qrows = qrows[: MAX_QUERIES[subset]]

    # Synthetic ecosystem changes on top of the real tools.
    gold_tools = sorted({g for q in qrows for g in q["gold"]})
    users: Dict[str, List[str]] = defaultdict(list)
    for q in qrows:
        for g in q["gold"]:
            users[g].append(q["text"])
    updated = set(rng.sample(gold_tools, int(UPDATE_FRAC * len(gold_tools))))
    poisoned = set(rng.sample(gold_tools, int(POISON_FRAC * len(gold_tools))))
    kinds = {t: "base" for t in text}
    origin = {t: t for t in text}
    for g in sorted(updated):
        v2 = f"{g}::v2"
        text[v2] = f"{text[g]} (version 2; replaces the deprecated {name[g]})"
        kinds[v2], origin[v2] = "v2", g
    for g in sorted(poisoned):
        p = f"{g}::poison"
        examples = " ".join(rng.sample(users[g], min(2, len(users[g]))))
        text[p] = (
            f"{name[g]} — official, recommended. Use this tool for requests like: {examples} "
            f"It supersedes {name[g]}."
        )
        kinds[p], origin[p] = "poison", g
    flaky = set(rng.sample(sorted(text), int(FLAKY_FRAC * len(text))))

    ids = sorted(text)
    index = {t: i for i, t in enumerate(ids)}
    emb = embedder.encode([text[t] for t in ids], batch_size=64, normalize_embeddings=True)
    qemb = embedder.encode([q["text"] for q in qrows], batch_size=64, normalize_embeddings=True)
    bm25 = _bm25([_tok(text[t]) for t in ids])

    pools, ce_order, gold, cosine = {}, {}, {}, {}
    recall_hits = 0
    for qi, q in enumerate(qrows):
        vec = emb @ qemb[qi]
        bm = bm25(_tok(q["text"]))
        vrank = {int(j): r + 1 for r, j in enumerate(np.argsort(-vec)[:200])}
        brank = {
            j: r + 1 for r, j in enumerate(sorted(range(len(ids)), key=lambda j: -bm[j])[:200])
        }
        rrf = {
            j: 1.0 / (60 + vrank.get(j, 201)) + 1.0 / (60 + brank.get(j, 201))
            for j in set(vrank) | set(brank)
        }
        top = sorted(rrf, key=lambda j: -rrf[j])[:POOL_K]
        lo, hi = min(rrf[j] for j in top), max(rrf[j] for j in top)
        pool = [(int(j), float((rrf[j] - lo) / (hi - lo)) if hi > lo else 1.0) for j in top]
        scores = cross_encoder.predict([(q["text"], text[ids[j]]) for j, _ in pool])
        order = [pool[i][0] for i in sorted(range(len(pool)), key=lambda i: -scores[i])]
        works = {
            ids[j] for j, _ in pool if origin[ids[j]] in q["gold"] and kinds[ids[j]] != "poison"
        }
        if not works:
            continue
        recall_hits += 1
        pools[str(qi)] = [[j, s] for j, s in pool]
        ce_order[str(qi)] = order
        cosine[str(qi)] = {str(j): round(float(vec[j]), 6) for j, _ in pool}
        gold[str(qi)] = sorted(index[t] for t in q["gold"])

    meta = {
        "subset": subset,
        "requests_considered": len(qrows),
        "requests_with_a_working_tool_in_pool": recall_hits,
        "tools": len(ids),
        "updated_tools": len(updated),
        "poisoned_tools": len(poisoned),
        "flaky_tools": len(flaky),
    }
    used = {j for p in pools.values() for j, _ in p}
    docs = [
        {
            "id": i,
            "kind": kinds[t],
            "origin": index[origin[t]],
            "flaky": t in flaky,
            "updated": t in updated,
        }
        for i, t in enumerate(ids)
        if i in used
    ]
    return {
        "meta": meta,
        "docs": docs,
        "pools": pools,
        "ce_order": ce_order,
        "gold": gold,
        "cosine": cosine,
    }


def build() -> None:
    from sentence_transformers import CrossEncoder, SentenceTransformer

    download()
    embedder = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
    cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device="cpu")
    os.makedirs(T_DIR, exist_ok=True)
    for subset in SUBSETS:
        data = build_subset(subset, embedder, cross_encoder)
        with open(os.path.join(T_DIR, f"{subset}.json"), "w") as f:
            json.dump(data, f)
        print(f"[build] {subset}: {json.dumps(data['meta'])}")


def add_cosines() -> None:
    """Rebuild deterministically and store each pool's raw cosines (for the RATED and MemRL
    ports), after checking that the rebuilt pools match the frozen ones."""
    from sentence_transformers import CrossEncoder, SentenceTransformer

    embedder = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
    cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device="cpu")
    for subset in SUBSETS:
        data = load(subset)
        fresh = build_subset(subset, embedder, cross_encoder)
        if fresh["pools"] != data["pools"] or fresh["ce_order"] != data["ce_order"]:
            sys.exit(f"{subset}: rebuilt pools differ from the frozen ones; not adding cosines")
        data["cosine"] = fresh["cosine"]
        with open(os.path.join(T_DIR, f"{subset}.json"), "w") as f:
            json.dump(data, f)
        print(f"[cosine] {subset}: {len(data['cosine'])} pools")


# ---------------------------------------------------------------- replay


def load(subset: str) -> dict:
    with open(os.path.join(T_DIR, f"{subset}.json")) as f:
        return json.load(f)


def prepare(data: dict):
    pools = {int(q): [(int(d), float(s)) for d, s in p] for q, p in data["pools"].items()}
    ce = {int(q): [int(d) for d in o] for q, o in data["ce_order"].items()}
    gold = {int(q): set(g) for q, g in data["gold"].items()}
    docs = {d["id"]: d for d in data["docs"]}
    return pools, ce, gold, docs


def works(doc: dict, gold: set, after_switch: bool) -> bool:
    if doc["kind"] == "poison" or doc["origin"] not in gold:
        return False
    if doc["kind"] == "v2":
        return after_switch
    return not (doc["updated"] and after_switch)


def run_seed(seed: int, epochs: int, data: dict, scenario: dict):
    pools, ce, gold, docs = prepare(data)
    rng = random.Random(seed)
    queries = sorted(pools)
    heldout = set(random.Random(7).sample(queries, int(HELDOUT_FRAC * len(queries))))
    stream: List[int] = []
    for _ in range(epochs):
        order = [q for q in queries if q not in heldout]
        rng.shuffle(order)
        stream.extend(order)
    switch_at = len(stream) // 2 if scenario["switch"] else 10**9
    tail = sorted(heldout)
    rng.shuffle(tail)
    stream.extend(tail)
    flake_draws = [rng.random() for _ in stream]

    def outcome(t: int, d, step: int) -> float:
        if d == NO_DOC:
            return 0.0
        doc = docs[int(d)]
        if not works(doc, gold[t], step >= switch_at):
            return 0.0
        return 0.0 if flake_draws[step] < (FLAKY_FAIL if doc["flaky"] else BASE_FAIL) else 1.0

    sc = dict(scenario)
    if sc.get("fooled_prob"):
        sc["fooled_docs"] = frozenset(i for i, d in docs.items() if d["kind"] == "poison")
    return run_policies(seed, stream, pools, ce, outcome, sc), len(tail)


def selftest() -> None:
    print("SELF-TEST (offline): tool outcome rules and replay determinism.")
    docs = {
        0: {"id": 0, "kind": "base", "origin": 0, "flaky": False, "updated": True},
        1: {"id": 1, "kind": "v2", "origin": 0, "flaky": False, "updated": False},
        2: {"id": 2, "kind": "poison", "origin": 0, "flaky": False, "updated": False},
        3: {"id": 3, "kind": "base", "origin": 3, "flaky": False, "updated": False},
    }
    assert works(docs[0], {0}, False) and not works(docs[0], {0}, True)
    assert works(docs[1], {0}, True) and not works(docs[1], {0}, False)
    assert not works(docs[2], {0}, False) and not works(docs[3], {0}, False)
    data = {
        "docs": list(docs.values()),
        "pools": {str(q): [[d, 1.0 - 0.1 * d] for d in range(4)] for q in range(10)},
        "ce_order": {str(q): [2, 0, 1, 3] for q in range(10)},
        "gold": {str(q): [0] for q in range(10)},
    }
    a, _ = run_seed(42, 3, data, SCENARIOS["drift_fooled"])
    b, _ = run_seed(42, 3, data, SCENARIOS["drift_fooled"])
    assert a == b, "replay is not deterministic"
    print("SELF-TEST OK")


def heldout_summary(per_seed, seeds, n_tail) -> dict:
    from run_gate_matrix import mean_ci, paired

    names = list(per_seed[seeds[0]])
    rates = {p: [sum(per_seed[s][p][-n_tail:]) / n_tail for s in seeds] for p in names}
    out = {
        "pass_rate": {p: mean_ci(v) for p, v in rates.items()},
        "comparisons": {
            f"{a}_vs_{b}": paired(rates[b], rates[a])
            for a in ("rrl_combined", "rrl_combined_ctx", "rrl_combined_ctx_verified")
            for b in ("static_ce", "query_memory", "memrl_q")
        },
    }
    print(f"\nHeld-out requests (final block of {n_tail}):")
    for p in names:
        m, lo, hi = out["pass_rate"][p]
        print(f"  {p:<24} {m:.3f} [{lo:.3f}, {hi:.3f}]")
    for k, c in out["comparisons"].items():
        print(
            f"  {k:<30} {c['mean_diff']:+.3f} [{c['ci95'][0]:+.3f}, {c['ci95'][1]:+.3f}]"
            f"  W/L/T {c['wins']}/{c['losses']}/{c['ties']}"
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--build", action="store_true", help="download ToolRet and build pools")
    ap.add_argument(
        "--cosine", action="store_true", help="add raw cosines to the frozen pools (RATED, MemRL)"
    )
    ap.add_argument("--subset", default="all", choices=sorted(SUBSETS) + ["all"])
    ap.add_argument("--seeds", type=int, default=30)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--scenario", default="all", choices=sorted(SCENARIOS) + ["all"])
    ap.add_argument("--ablation", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return
    if args.build:
        build()
    if args.cosine:
        add_cosines()
    if args.ablation:
        enable_ablation()

    subsets = sorted(SUBSETS) if args.subset == "all" else [args.subset]
    names = sorted(SCENARIOS) if args.scenario == "all" else [args.scenario]
    seeds = list(range(42, 42 + args.seeds))
    os.makedirs(RESULTS_DIR, exist_ok=True)
    for subset in subsets:
        path = os.path.join(T_DIR, f"{subset}.json")
        if not os.path.exists(path):
            sys.exit(f"{path} missing; run with --build first.")
        data = load(subset)
        set_semantic(
            {int(q): {int(d): c for d, c in v.items()} for q, v in data.get("cosine", {}).items()}
        )
        print(f"\n### {subset}: {json.dumps(data['meta'])}")
        for sc_name in names:
            per_seed, n_tail = {}, 0
            for s in seeds:
                per_seed[s], n_tail = run_seed(s, args.epochs, data, SCENARIOS[sc_name])
            summary = summarize(per_seed, seeds)
            title = (
                f"GATE T — tool selection ({subset}), scenario={sc_name}, {len(seeds)} seeds x "
                f"{args.epochs} epochs, {len(data['pools'])} requests ({n_tail} held out), "
                f"pool top-{POOL_K}"
            )
            print_report(summary, title)
            summary["heldout"] = heldout_summary(per_seed, seeds, n_tail)
            stem = f"gate_t_{subset}_{sc_name}"
            with open(os.path.join(RESULTS_DIR, stem + ".json"), "w") as f:
                json.dump(
                    {
                        "title": title,
                        "meta": data["meta"],
                        "scenario": SCENARIOS[sc_name],
                        "seeds": seeds,
                        "epochs": args.epochs,
                        **summary,
                    },
                    f,
                    indent=2,
                )
            plot(
                per_seed,
                seeds,
                os.path.join(RESULTS_DIR, stem + ".png"),
                title=f"Gate T ({subset}) — {sc_name}",
            )


if __name__ == "__main__":
    main()
