"""
Gate V: tool selection on MCPEvol-Bench toolsets (arXiv 2607.14642;
github.com/Octobrist/MCPEvol-Bench).

MCPEvol-Bench evaluates LLM agents on 393 tasks over real MCP servers whose toolsets evolve in
four steps: a tool's description is rewritten (DESC), optional parameters are added (PARAM), or
a new tool is added to the server (TOOL). Its own score is an LLM judge's rating of full agent
runs on the live servers, which a selection-layer replay cannot reproduce. This gate derives a
tool-selection task from its data, unmodified:

  request      (task, server): MCPEvol's task text and one server it uses
  candidates   that server's tools at the current stage (original descriptions from
               status_dict.json, rewritten descriptions and added tools from the evolution
               files), top 20 by MiniLM similarity
  correct      any tool the task's successful reference run called on that server
  stages       the requests replay three times, on the early (original), middle (after two
               evolution steps) and late (after four) toolsets; relevance is recomputed from
               each stage's descriptions, and a tool keeps its id across stages
Added tools are scored against the original reference, so they count as wrong even where they
might also serve. Feedback: as in Gate X, every executed call reports success, so only verified
calls carry information. That feedback model is ours.

MCPEvol-Bench is not redistributed here (its repository has no licence). --build reads
src/custom/data from --mcpevol-dir and writes only ids, tool names and scores to data/gate_v/.

    python3 sim/run_gate_v.py --build --mcpevol-dir PATH/TO/src/custom/data
    python3 sim/run_gate_v.py --scenario all
"""

import argparse
import json
import os
import random
import sys
from collections import Counter
from typing import Dict, List, Tuple

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rrl.feedback import OutcomeSignals  # noqa: E402
from run_gate_matrix import (  # noqa: E402
    RESULTS_DIR,
    RUN_POLICIES,
    enable_ablation,
    make_policy,
    mean_ci,
    plot,
    print_report,
    summarize,
)

HERE = os.path.dirname(os.path.abspath(__file__))
V_DIR = os.path.abspath(os.path.join(HERE, "..", "data", "gate_v"))
MCPEVOL_COMMIT = "888c779a0fe735f612ca62bdd1fdd7c9e6bf8701"
LEVELS = ["HYBRID", "HYBRID/HYBRID", "HYBRID/HYBRID/HYBRID", "HYBRID/HYBRID/HYBRID/HYBRID"]
STAGES = {"early": 0, "middle": 2, "late": 4}  # evolution steps applied
POOL_K = 20
HELDOUT_FRAC = 0.2
SCENARIOS: Dict[str, dict] = {
    "stationary": {"coverage": 1.0},
    "spot_check": {"coverage": 0.2},
}


def toolsets(data_dir: str) -> Tuple[Dict[str, List[Tuple[str, str]]], List[dict]]:
    """Original toolsets and the four evolution steps, as {server: [(name, description)]}."""
    with open(os.path.join(data_dir, "test_cases", "status_dict.json")) as f:
        status = json.load(f)
    base = {
        s: [(t["name"], t.get("description") or "") for t in v["tools"]] for s, v in status.items()
    }
    steps = []
    for level in LEVELS:
        with open(os.path.join(data_dir, "mutation", level, "evolution.json")) as f:
            steps.append(json.load(f))
    return base, steps


def stage_tools(base, steps, server: str, n_steps: int) -> List[Tuple[str, str]]:
    """The server's tools after the first n_steps evolution steps."""
    tools = dict(base.get(server, []))
    for ev in steps[:n_steps]:
        m = ev.get(server)
        if not m:
            continue
        if m["mutation_type"] == "DESC":
            for name, v in m.items():
                if isinstance(v, dict) and name in tools:
                    tools[name] = v.get("tool", {}).get("modified_description") or tools[name]
        elif m["mutation_type"] == "TOOL":
            cfg = m.get("new_tool_config") or {}
            if cfg.get("name"):
                tools[cfg["name"]] = cfg.get("description") or ""
    return list(tools.items())


# ---------------------------------------------------------------- build


def build(data_dir: str) -> None:
    from sentence_transformers import CrossEncoder, SentenceTransformer

    with open(os.path.join(data_dir, "test_cases", "test_cases.json")) as f:
        cases = json.load(f)
    base, steps = toolsets(data_dir)
    embedder = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device="cpu")

    tool_ids: Dict[Tuple[str, str], int] = {}
    requests: List[dict] = []
    dropped: Counter = Counter()
    for case in cases:
        if not (case.get("eval_result") or {}).get("is_success"):
            dropped["reference run not successful"] += 1
            continue
        for server, used in case["servers"].items():
            if server not in base:
                dropped["server missing from status_dict"] += 1
                continue
            names = {n for n, _ in base[server]}
            correct = sorted(set(used) & names)
            if not correct:
                dropped["reference tool not in the server's toolset"] += 1
                continue
            requests.append(
                {
                    "id": len(requests),
                    "task": case["idx"],
                    "server": server,
                    "question": case["question"],
                    "correct": correct,
                }
            )

    pending = []  # (request, stage, [(tool id, text)])
    for r in requests:
        r["stages"] = {}
        for stage, n in STAGES.items():
            cands = []
            for name, desc in stage_tools(base, steps, r["server"], n):
                tid = tool_ids.setdefault((r["server"], name), len(tool_ids))
                cands.append((tid, f"{name}: {desc}"))
            pending.append((r, stage, cands))
    q_emb = embedder.encode(
        [r["question"] for r, _, _ in pending], normalize_embeddings=True, batch_size=32
    )
    flat = [txt for _, _, cands in pending for _, txt in cands]
    t_emb = embedder.encode(flat, normalize_embeddings=True, batch_size=64)
    k = 0
    shortlist = []
    for n, (r, stage, cands) in enumerate(pending):
        sims = []
        for tid, txt in cands:
            sims.append((tid, round(float(q_emb[n] @ t_emb[k]), 6), txt))
            k += 1
        top = sorted(sims, key=lambda p: -p[1])[:POOL_K]
        shortlist.append((r, stage, top))
    ce = reranker.predict(
        [(r["question"], txt) for r, _, top in shortlist for _, _, txt in top], batch_size=64
    )
    k = 0
    for r, stage, top in shortlist:
        scored = []
        for tid, _, _ in top:
            scored.append((tid, float(ce[k])))
            k += 1
        r["stages"][stage] = {
            "pool": [(tid, s) for tid, s, _ in top],
            "ce": [tid for tid, _ in sorted(scored, key=lambda p: -p[1])],
        }

    names_by_id = {v: list(k) for k, v in tool_ids.items()}
    os.makedirs(V_DIR, exist_ok=True)
    with open(os.path.join(V_DIR, "tools.json"), "w") as f:
        json.dump([names_by_id[i] for i in range(len(names_by_id))], f)
    with open(os.path.join(V_DIR, "requests.jsonl"), "w") as f:
        for r in requests:
            out = {key: r[key] for key in ("id", "task", "server", "stages")}
            out["correct"] = [tool_ids[(r["server"], n)] for n in r["correct"]]
            f.write(json.dumps(out) + "\n")
    meta = {
        "source": "Octobrist/MCPEvol-Bench src/custom/data",
        "commit": MCPEVOL_COMMIT,
        "requests": len(requests),
        "tasks": len({r["task"] for r in requests}),
        "dropped": dict(dropped),
        "stages": STAGES,
        "pool_k": POOL_K,
        "changed_pools": sum(
            r["stages"]["early"]["ce"] != r["stages"]["late"]["ce"] for r in requests
        ),
    }
    with open(os.path.join(V_DIR, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps(meta, indent=2))


# ---------------------------------------------------------------- replay


class Env:
    def __init__(self) -> None:
        with open(os.path.join(V_DIR, "requests.jsonl")) as f:
            self.req = [json.loads(line) for line in f]
        for r in self.req:
            r["correct"] = set(r["correct"])
            for st in r["stages"].values():
                st["pool"] = [(d, s) for d, s in st["pool"]]
        self.pools = {(r["id"], s): v["pool"] for r in self.req for s, v in r["stages"].items()}
        tasks = sorted({r["task"] for r in self.req})
        held = set(random.Random(7).sample(tasks, int(HELDOUT_FRAC * len(tasks))))
        self.held = held


def stream_for(env: Env, seed: int) -> Tuple[List[Tuple[int, str]], int]:
    """Each seen request once per stage (early, middle, late), then the held-out tasks' requests
    on the late toolset."""
    rng = random.Random(seed)
    seen = [r["id"] for r in env.req if r["task"] not in env.held]
    held = [r["id"] for r in env.req if r["task"] in env.held]
    stream: List[Tuple[int, str]] = []
    for stage in STAGES:
        order = list(seen)
        rng.shuffle(order)
        stream.extend((i, stage) for i in order)
    rng.shuffle(held)
    return stream + [(i, "late") for i in held], len(held)


def replay_seed(env: Env, seed: int, scenario: dict, names: List[str]):
    stream, n_held = stream_for(env, seed)
    rng = random.Random(seed * 7919 + 5)
    cov = [rng.random() < scenario["coverage"] for _ in stream]
    hist: Dict[str, List[float]] = {}
    for name in names:
        if name == "oracle":
            hist[name] = [
                float(any(d in env.req[i]["correct"] for d, _ in env.req[i]["stages"][st]["pool"]))
                for i, st in stream
            ]
            continue
        ce_order: Dict[int, List[int]] = {}
        choose, update = make_policy(name, seed, ce_order, env.pools)
        h = []
        for step, (i, stage) in enumerate(stream):
            r = env.req[i]
            st = r["stages"][stage]
            ce_order[i] = st["ce"]
            d = choose(i, st["pool"], step)
            y = 1.0 if d in r["correct"] else 0.0
            # Every executed MCP call reports success and is well formed.
            update(
                i, d, OutcomeSignals(s_gt=y if cov[step] else None, s_behave=1.0, s_judge=1.0), step
            )
            h.append(y)
        hist[name] = h
    return hist, n_held


def report(per_seed, seeds, n_held, title, stem, extra) -> None:
    summary = summarize(per_seed, seeds)
    print_report(summary, title)
    names = list(per_seed[seeds[0]])
    held = {p: mean_ci([sum(per_seed[s][p][-n_held:]) / n_held for s in seeds]) for p in names}
    print(f"\nHeld-out tasks (final block of {n_held}):")
    for p in names:
        print(f"  {p:<32} {held[p][0]:.3f} [{held[p][1]:.3f}, {held[p][2]:.3f}]")
    summary["heldout"] = {"pass_rate": held}
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, stem + ".json"), "w") as f:
        json.dump({"title": title, **extra, "seeds": seeds, **summary}, f, indent=2)
    plot(per_seed, seeds, os.path.join(RESULTS_DIR, stem + ".png"), title=title.split(",")[0])


# ---------------------------------------------------------------- self-test


def selftest() -> None:
    print("SELF-TEST: MCPEvol toolset evolution and replay.")
    base = {"srv": [("a", "first"), ("b", "second")]}
    steps = [
        {"srv": {"mutation_type": "DESC", "a": {"tool": {"modified_description": "rewritten"}}}},
        {"srv": {"mutation_type": "TOOL", "new_tool_config": {"name": "c", "description": "new"}}},
        {"srv": {"mutation_type": "PARAM", "b": {"evolution_summary": "x"}}},
    ]
    assert stage_tools(base, steps, "srv", 0) == [("a", "first"), ("b", "second")]
    assert stage_tools(base, steps, "srv", 1) == [("a", "rewritten"), ("b", "second")]
    assert stage_tools(base, steps, "srv", 3) == [("a", "rewritten"), ("b", "second"), ("c", "new")]
    if os.path.exists(os.path.join(V_DIR, "requests.jsonl")):
        env = Env()
        for r in env.req:
            assert set(r["stages"]) == set(STAGES)
            for st in r["stages"].values():
                assert {d for d, _ in st["pool"]} == set(st["ce"]) and len(st["pool"]) <= POOL_K
        hist, n_held = replay_seed(
            env, 42, SCENARIOS["spot_check"], ["static_ce", "rrl_combined", "oracle"]
        )
        assert len({len(h) for h in hist.values()}) == 1 and 0 < n_held < len(hist["oracle"])
    print("SELF-TEST OK")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--build", action="store_true", help="derive data/gate_v from MCPEvol-Bench")
    ap.add_argument("--mcpevol-dir", default=".", help="MCPEvol-Bench's src/custom/data folder")
    ap.add_argument("--seeds", type=int, default=30)
    ap.add_argument("--scenario", default="all", choices=sorted(SCENARIOS) + ["all"])
    ap.add_argument("--ablation", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        selftest()
        return
    if args.build:
        build(args.mcpevol_dir)
    if args.ablation:
        enable_ablation()
    env = Env()
    names = list(RUN_POLICIES)
    seeds = list(range(42, 42 + args.seeds))
    with open(os.path.join(V_DIR, "meta.json")) as f:
        meta = json.load(f)
    for sc in sorted(SCENARIOS) if args.scenario == "all" else [args.scenario]:
        per_seed = {}
        n_held = 0
        for sd in seeds:
            per_seed[sd], n_held = replay_seed(env, sd, SCENARIOS[sc], names)
        title = (
            f"GATE V — tool selection on MCPEvol-Bench toolsets, scenario={sc}, {len(seeds)} "
            f"seeds, {len(env.req)} requests x 3 stages"
        )
        report(
            per_seed,
            seeds,
            n_held,
            title,
            f"gate_v_mcpevol_{sc}",
            {"scenario": SCENARIOS[sc], "data": meta},
        )


if __name__ == "__main__":
    main()
