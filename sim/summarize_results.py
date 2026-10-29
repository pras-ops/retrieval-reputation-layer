"""
Print the markdown result tables in RESULTS.md from the JSON files in sim/results/.

    python3 sim/summarize_results.py > /tmp/tables.md
"""

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")

MAIN = [
    ("static_ce", "static reranker"),
    ("query_memory", "reuse what worked"),
    ("query_memory_robust", "reuse what worked, robust"),
    ("memrl_q", "MemRL-style"),
    ("rrl_combined", "RRL combined"),
    ("rrl_combined_e10", "RRL combined, explore 0.10"),
    ("rrl_combined_ctx", "RRL combined + per-context"),
    ("rrl_combined_ctx_verified", "RRL + per-context + verified memory"),
    ("rrl_combined_ctx_verified_fixed", "same, fixed feedback rule instead of learned trust"),
    ("oracle", "best possible"),
]
ABLATION = [
    ("rrl_ce", "RRL, reranker input only"),
    ("rrl_combined_globalrel", "combined, global-only reliability"),
    ("rrl_combined_nocd", "combined, no change detection"),
    ("rrl_combined_cdold", "combined, old detector (w=4, 0.6)"),
    ("rrl_combined_e0", "combined, exploration off"),
    ("rrl_combined_conf", "combined, confidence-gated exploration"),
    ("rrl_combined_ctx_conf", "per-context + confidence-gated exploration"),
]

MCP = [
    ("static_ce", "static reranker"),
    ("llm_picks", "Gemini picks from tool list"),
    ("query_memory", "reuse what worked"),
    ("query_memory_robust", "reuse what worked, robust"),
    ("memrl_q", "MemRL-style"),
    ("rrl_combined", "RRL combined"),
    ("rrl_combined_ctx", "RRL combined + per-context"),
    ("rrl_combined_ctx_verified", "RRL + per-context + verified memory"),
    ("rrl_combined_ctx_verified_fixed", "same, fixed feedback rule instead of learned trust"),
    ("oracle", "best possible"),
]

GATES = [
    (
        "gate_t_toolbench",
        "Gate T — tool selection, ToolBench (ToolRet)",
        ["stationary", "sparse_verifier", "sycophantic", "fooled", "drift", "drift_fooled"],
    ),
    (
        "gate_t_toollens",
        "Gate T — tool selection, ToolLens (ToolRet)",
        ["stationary", "sparse_verifier", "sycophantic", "fooled", "drift", "drift_fooled"],
    ),
    (
        "gate_w",
        "Gate W — web-scraping extractor selection",
        ["stationary", "spot_check", "redesign", "redesign_spot_check"],
    ),
    (
        "gate_m_replay",
        "Gate M — live MCP servers (replayed), with the poisoned server",
        ["stationary", "spot_check", "update", "update_spot_check"],
    ),
    (
        "gate_m_replay",
        "Gate M — staged defection, with the poisoned server",
        ["defect", "defect_spot_check"],
    ),
    (
        "gate_m_replay_clean",
        "Gate M — live MCP servers (replayed), clean registry",
        ["stationary", "spot_check", "update", "update_spot_check"],
    ),
    (
        "gate_m_replay_clean",
        "Gate M — staged defection, clean registry",
        ["defect", "defect_spot_check"],
    ),
    (
        "gate_g_hard",
        "Gate G hard — private-API docs, poisoned reranker",
        ["stationary", "sycophantic", "fooled", "drift", "drift_sycophantic", "drift_fooled"],
    ),
]


def cell(d: dict, p: str, heldout: bool) -> str:
    o = d["overall"].get(p)
    if o is None:
        return "—"
    s = f"{o[0] * 100:.1f}"
    h = d.get("heldout", {}).get("pass_rate", {}).get(p) if heldout else None
    return f"{s} / {h[0] * 100:.1f}" if h else s


def table(stem: str, title: str, scen: list, rows: list) -> str:
    data = {}
    for s in scen:
        path = os.path.join(RES, f"{stem}_{s}.json")
        if os.path.exists(path):
            with open(path) as f:
                data[s] = json.load(f)
    if not data:
        return ""
    held = any("heldout" in d for d in data.values())
    out = [f"### {title}", ""]
    if held:
        out.append("Overall / held-out pass rate (%), mean of 30 seeds.\n")
    out.append("| Policy | " + " | ".join(data) + " |")
    out.append("|---|" + "---|" * len(data))
    for p, label in rows:
        out.append(f"| {label} | " + " | ".join(cell(d, p, held) for d in data.values()) + " |")
    return "\n".join(out) + "\n"


def main() -> None:
    for stem, title, scen in GATES:
        print(table(stem, title, scen, MCP if stem.startswith("gate_m") else MAIN))
        print(table(stem, title + " — ablation", scen, ABLATION))


if __name__ == "__main__":
    main()
