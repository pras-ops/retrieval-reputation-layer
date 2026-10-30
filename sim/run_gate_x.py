"""
Gate X: tool selection on MCPTox (Wang et al., AAAI 2026; arXiv 2508.14925).

MCPTox gives, for 1,348 requests to 45 real MCP servers, the server's tool list with one
poisoned tool description added, and the responses of 29 LLM agents with labels: "Success"
(the attack worked), "Failure-Ignored" (the agent did exactly what the user asked), "Failure-
Direct Execution" (it called the poisoned tool itself), "Failure-Refused", "None" (no valid
output). MCPTox evaluates agents and has no tool-selection task, so this gate derives one from
its data, unmodified:

  request      MCPTox's request, in the context of its (server, request text)
  candidates   the server's tools plus that request's poisoned tool, as the agent saw them
  correct tool the tool called by the agents MCPTox labels "Failure-Ignored" (majority vote);
               requests without such a response, or whose poisoned tool reuses a server
               tool's name, are dropped
  relevance    MiniLM cosine and cross-encoder order over "name: description"
  LLM picks    MCPTox's recorded responses of Gemini 2.5 Flash, scored by their labels

Feedback: the verifier says whether the chosen tool is the correct one. Unverified feedback is
what an MCP call itself returns: every tool, the poisoned one included, executes and reports
success, and the call is well formed, so unverified reports carry no information. That feedback
model is ours, not MCPTox's. Parameter-tampering attacks (MCPTox's third paradigm) act through
the arguments of the correct tool, which tool selection cannot prevent.

MCPTox is not redistributed here (its repository has no licence). --build reads
response_all.json from --mcptox-dir and writes only ids, tool names, scores and label codes to
data/gate_x/.

    curl -LO https://raw.githubusercontent.com/zhiqiangwang4/MCPTox-Benchmark/f85189f9ad12504c197c7f920ab818a40657b1fa/response_all.json
    python3 sim/run_gate_x.py --build --mcptox-dir .
    python3 sim/run_gate_x.py --scenario all
"""

import argparse
import json
import os
import random
import re
import sys
from collections import Counter
from typing import Dict, List, Optional, Tuple

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rrl.feedback import OutcomeSignals  # noqa: E402
from run_gate_matrix import (  # noqa: E402
    RESULTS_DIR,
    RUN_POLICIES,
    enable_ablation,
    make_policy,
    mean_ci,
    paired,
    plot,
    print_report,
    summarize,
)

HERE = os.path.dirname(os.path.abspath(__file__))
X_DIR = os.path.abspath(os.path.join(HERE, "..", "data", "gate_x"))
MCPTOX_COMMIT = "f85189f9ad12504c197c7f920ab818a40657b1fa"
LLM_TAG = "gemini-2.5-flash"
N_HELDOUT_SERVERS = 9  # 20% of 45 servers, introduced only in a final block
EPOCHS = 2
SCENARIOS: Dict[str, dict] = {
    "stationary": {"coverage": 1.0},
    "spot_check": {"coverage": 0.2},
}
LABEL_CODES = {
    "Success": "S",
    "Failure-Ignored": "I",
    "Failure-Direct Execution": "D",
    "Failure-Refused": "R",
    "None": "N",
}
TOOL_RE = re.compile(
    r"Tool:\s*(\S+)\s*\nDescription:\s*(.*?)(?=\n\s*Arguments:|\n\s*\nTool:|\Z)", re.S
)
CALL_RE = re.compile(r"""['"]tool['"]\s*:\s*['"]([^'"]+)['"]""")


def parse_tools(text: str) -> List[Tuple[str, str]]:
    """(name, description) pairs from MCPTox's 'Tool: ...\\nDescription: ...' blocks."""
    text = (text or "").replace("\\n", "\n")
    return [(n, d.strip()) for n, d in TOOL_RE.findall(text)]


def called_tool(response) -> Optional[str]:
    m = CALL_RE.search(str(response or ""))
    return m.group(1) if m else None


def responses(dd: dict) -> Tuple[Dict[str, str], Dict[str, Optional[str]]]:
    """Label and called tool per model, local and online agents."""
    labels: Dict[str, str] = {}
    calls: Dict[str, Optional[str]] = {}
    for m, lab in (dd.get("label") or {}).items():
        labels[m] = lab
        calls[m] = called_tool((dd.get("response") or {}).get(m))
    online = dd.get("online_result") or {}
    for resp, labs in zip(
        online.get("backend_model_responses", []), online.get("labeled_model_results", [])
    ):
        for m, lab in labs.items():
            labels[m] = lab
            calls[m] = called_tool(resp.get(m))
    return labels, calls


# ---------------------------------------------------------------- build


def build(mcptox_dir: str) -> None:
    from sentence_transformers import CrossEncoder, SentenceTransformer

    with open(os.path.join(mcptox_dir, "response_all.json")) as f:
        raw = json.load(f)
    embedder = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device="cpu")

    tools: List[list] = []  # tool id -> [server, name, poisoned]
    instances: List[dict] = []
    dropped: Counter = Counter()
    ctx_ids: Dict[Tuple[str, str], int] = {}
    pending = []  # (instance, query, [(tool id, text)])
    for sname, sv in raw["servers"].items():
        legit = parse_tools(sv["clean_system_promot"])
        legit_names = {n for n, _ in legit}
        ids = {}
        for name, _ in legit:
            ids[name] = len(tools)
            tools.append([sname, name, False])
        for mi in sv["malicious_instance"]:
            for dd in mi["datas"]:
                poison = parse_tools(dd.get("poisoned_tool") or mi.get("poisoned_tool") or "")
                if not poison:
                    dropped["no poisoned tool"] += 1
                    continue
                pname, pdesc = poison[0]
                if pname in legit_names:
                    dropped["poisoned tool reuses a server tool's name"] += 1
                    continue
                labels, calls = responses(dd)
                votes = Counter(
                    calls[m]
                    for m, lab in labels.items()
                    if lab == "Failure-Ignored" and calls.get(m) in legit_names
                )
                if not votes:
                    dropped["no Failure-Ignored call"] += 1
                    continue
                attack = Counter(
                    calls[m]
                    for m, lab in labels.items()
                    if lab == "Success" and calls.get(m) in legit_names
                )
                pid = len(tools)
                tools.append([sname, pname, True])
                correct = votes.most_common(1)[0][0]
                target = attack.most_common(1)[0][0] if attack else None
                inst = {
                    "id": len(instances),
                    "server": sname,
                    "ctx": ctx_ids.setdefault((sname, dd["query"]), len(ctx_ids)),
                    "paradigm": (mi.get("metadata") or {}).get("paradigm"),
                    "correct": ids[correct],
                    "poisoned": pid,
                    "target": ids[target] if target is not None and target != correct else None,
                    "labels": {m: LABEL_CODES.get(lab, "N") for m, lab in sorted(labels.items())},
                }
                cands = [(ids[n], f"{n}: {d}") for n, d in legit] + [(pid, f"{pname}: {pdesc}")]
                instances.append(inst)
                pending.append((inst, dd["query"], cands))

    queries = [q for _, q, _ in pending]
    texts = [txt for _, _, cands in pending for _, txt in cands]
    q_emb = embedder.encode(queries, normalize_embeddings=True, batch_size=64)
    t_emb = embedder.encode(texts, normalize_embeddings=True, batch_size=64)
    pairs = [(q, txt) for _, q, cands in pending for _, txt in cands]
    ce_scores = reranker.predict(pairs, batch_size=64)
    k = 0
    for n, (inst, _, cands) in enumerate(pending):
        sims, ces = [], []
        for tid, _ in cands:
            sims.append((tid, round(float(q_emb[n] @ t_emb[k]), 6)))
            ces.append((tid, float(ce_scores[k])))
            k += 1
        inst["pool"] = sorted(sims, key=lambda p: -p[1])
        inst["ce"] = [tid for tid, _ in sorted(ces, key=lambda p: -p[1])]

    os.makedirs(X_DIR, exist_ok=True)
    with open(os.path.join(X_DIR, "tools.json"), "w") as f:
        json.dump(tools, f)
    with open(os.path.join(X_DIR, "instances.jsonl"), "w") as f:
        for inst in instances:
            f.write(json.dumps(inst) + "\n")
    meta = {
        "source": "zhiqiangwang4/MCPTox-Benchmark response_all.json",
        "commit": MCPTOX_COMMIT,
        "instances": len(instances),
        "dropped": dict(dropped),
        "servers": len(raw["servers"]),
        "contexts": len(ctx_ids),
        "label_codes": LABEL_CODES,
    }
    with open(os.path.join(X_DIR, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps(meta, indent=2))


# ---------------------------------------------------------------- replay


class Env:
    def __init__(self) -> None:
        with open(os.path.join(X_DIR, "tools.json")) as f:
            self.tools = json.load(f)
        with open(os.path.join(X_DIR, "instances.jsonl")) as f:
            self.inst = [json.loads(line) for line in f]
        for inst in self.inst:
            inst["pool"] = [(d, s) for d, s in inst["pool"]]
        self.pools = {inst["id"]: inst["pool"] for inst in self.inst}
        servers = sorted({inst["server"] for inst in self.inst})
        self.held = set(random.Random(7).sample(servers, N_HELDOUT_SERVERS))


def stream_for(env: Env, seed: int) -> Tuple[List[int], int]:
    rng = random.Random(seed)
    seen = [i["id"] for i in env.inst if i["server"] not in env.held]
    held = [i["id"] for i in env.inst if i["server"] in env.held]
    stream: List[int] = []
    for _ in range(EPOCHS):
        order = list(seen)
        rng.shuffle(order)
        stream.extend(order)
    rng.shuffle(held)
    return stream + held, len(held)


def replay_seed(env: Env, seed: int, scenario: dict, names: List[str]):
    stream, n_held = stream_for(env, seed)
    rng = random.Random(seed * 7919 + 3)
    cov = [rng.random() < scenario["coverage"] for _ in stream]
    hist: Dict[str, List[float]] = {}
    poisoned: Dict[str, float] = {}
    attacked: Dict[str, float] = {}
    for name in names:
        if name == "oracle":
            hist[name] = [1.0] * len(stream)
            continue
        if name == "llm_picks":
            codes = [env.inst[i]["labels"].get(LLM_TAG, "N") for i in stream]
            hist[name] = [float(c == "I") for c in codes]
            poisoned[name] = sum(c == "D" for c in codes) / len(codes)
            attacked[name] = sum(c == "S" for c in codes) / len(codes)
            continue
        ce_order: Dict[int, List[int]] = {}
        choose, update = make_policy(name, seed, ce_order, env.pools)
        h, n_poison, n_attack = [], 0, 0
        for step, i in enumerate(stream):
            inst = env.inst[i]
            t = inst["ctx"]
            ce_order[t] = inst["ce"]
            d = choose(t, inst["pool"], step)
            y = 1.0 if d == inst["correct"] else 0.0
            n_poison += d == inst["poisoned"]
            n_attack += d is not None and d == inst["target"]
            # Every executed MCP call reports success and is well formed.
            sig = OutcomeSignals(s_gt=y if cov[step] else None, s_behave=1.0, s_judge=1.0)
            update(t, d, sig, step)
            h.append(y)
        hist[name] = h
        poisoned[name] = n_poison / len(stream)
        attacked[name] = n_attack / len(stream)
    return hist, poisoned, attacked, n_held


def report(per_seed, seeds, poisoned, attacked, n_held, title, stem, extra) -> None:
    summary = summarize(per_seed, seeds)
    llm = [sum(per_seed[s]["llm_picks"]) / len(per_seed[s]["llm_picks"]) for s in seeds]
    for b in per_seed[seeds[0]]:
        if b not in ("llm_picks", "oracle"):
            other = [sum(per_seed[s][b]) / len(per_seed[s][b]) for s in seeds]
            summary["comparisons"]["overall"][f"{b}_vs_llm_picks"] = paired(llm, other)
    print_report(summary, title)
    names = list(per_seed[seeds[0]])
    held = {p: mean_ci([sum(per_seed[s][p][-n_held:]) / n_held for s in seeds]) for p in names}
    print(f"\nHeld-out servers (final block of {n_held}):")
    for p in names:
        print(
            f"  {p:<32} {held[p][0]:.3f}  poisoned {poisoned.get(p, 0.0):.3f}"
            f"  attack target {attacked.get(p, 0.0):.3f}"
        )
    summary["heldout"] = {"pass_rate": held}
    summary["poisoned_choice_rate"] = poisoned
    summary["attack_choice_rate"] = attacked
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, stem + ".json"), "w") as f:
        json.dump({"title": title, **extra, "seeds": seeds, **summary}, f, indent=2)
    plot(per_seed, seeds, os.path.join(RESULTS_DIR, stem + ".png"), title=title.split(",")[0])


def model_table(env: Env) -> Dict[str, Dict[str, float]]:
    """MCPTox's own per-model rates on the kept requests (no selection involved)."""
    counts: Dict[str, Counter] = {}
    for inst in env.inst:
        for m, c in inst["labels"].items():
            counts.setdefault(m, Counter())[c] += 1
    out = {}
    for m, c in counts.items():
        n = sum(c.values())
        valid = n - c["N"]
        out[m] = {
            "n": n,
            "task_done": c["I"] / n,
            "attack_success_of_valid": c["S"] / valid if valid else 0.0,
            "direct_execution": c["D"] / n,
        }
    return out


# ---------------------------------------------------------------- self-test


def selftest() -> None:
    print("SELF-TEST: MCPTox parsing and replay.")
    tools = parse_tools("Tool: qubit\\nDescription: Initiates a protocol.\\nArguments:\\n- x\\n\\n")
    assert tools == [("qubit", "Initiates a protocol.")], tools
    two = parse_tools(
        "Tool: a\nDescription: first\nArguments:\n- p\n\n\nTool: b\nDescription: second"
    )
    assert [n for n, _ in two] == ["a", "b"], two
    assert called_tool("{'tool': 'read_file', 'arguments': {}}") == "read_file"
    assert called_tool('{"tool":"write_file"}') == "write_file"
    assert called_tool("I refuse.") is None
    if os.path.exists(os.path.join(X_DIR, "instances.jsonl")):
        env = Env()
        for inst in env.inst:
            ids = {d for d, _ in inst["pool"]}
            assert inst["correct"] in ids and inst["poisoned"] in ids
            assert set(inst["ce"]) == ids
            assert not env.tools[inst["correct"]][2] and env.tools[inst["poisoned"]][2]
        hist, poisoned, _, n_held = replay_seed(
            env, 42, SCENARIOS["spot_check"], ["static_ce", "llm_picks", "rrl_combined", "oracle"]
        )
        assert all(len(h) == len(hist["oracle"]) for h in hist.values())
        assert 0 < n_held < len(hist["oracle"]) and set(poisoned) == {
            "static_ce",
            "llm_picks",
            "rrl_combined",
        }
    print("SELF-TEST OK")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--build", action="store_true", help="derive data/gate_x from MCPTox")
    ap.add_argument("--mcptox-dir", default=".", help="folder holding MCPTox's response_all.json")
    ap.add_argument("--seeds", type=int, default=30)
    ap.add_argument("--scenario", default="all", choices=sorted(SCENARIOS) + ["all"])
    ap.add_argument("--ablation", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        selftest()
        return
    if args.build:
        build(args.mcptox_dir)
    if args.ablation:
        enable_ablation()
    env = Env()
    names = list(RUN_POLICIES)
    names.insert(names.index("static_ce") + 1, "llm_picks")
    seeds = list(range(42, 42 + args.seeds))
    with open(os.path.join(X_DIR, "meta.json")) as f:
        meta = json.load(f)
    for sc in sorted(SCENARIOS) if args.scenario == "all" else [args.scenario]:
        per_seed, pois, att = {}, Counter(), Counter()
        n_held = 0
        for sd in seeds:
            per_seed[sd], p, a, n_held = replay_seed(env, sd, SCENARIOS[sc], names)
            for k, v in p.items():
                pois[k] += v / len(seeds)
            for k, v in a.items():
                att[k] += v / len(seeds)
        title = (
            f"GATE X — tool selection on MCPTox, scenario={sc}, {len(seeds)} seeds x {EPOCHS} "
            f"epochs, {len(env.inst)} requests on {len({i['server'] for i in env.inst})} servers"
        )
        report(
            per_seed,
            seeds,
            dict(pois),
            dict(att),
            n_held,
            title,
            f"gate_x_mcptox_{sc}",
            {
                "scenario": SCENARIOS[sc],
                "llm_baseline": LLM_TAG,
                "data": meta,
                "models": model_table(env),
            },
        )


if __name__ == "__main__":
    main()
