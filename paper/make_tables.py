"""
Generate the paper's LaTeX tables and number macros from sim/results/*.json, so no number in the
paper is typed by hand.

    python3 paper/make_tables.py      # writes paper/tables/*.tex and paper/tables/numbers.tex
"""

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.abspath(os.path.join(HERE, "..", "sim", "results"))
OUT = os.path.join(HERE, "tables")

LABEL = {
    "static_ce": "Static reranker",
    "llm_picks": "LLM picks from tool list",
    "query_memory": "Per-context memory",
    "query_memory_robust": "Per-context memory, robust",
    "memrl_q": "MemRL-style",
    "rrl_combined": "RRL (global)",
    "rrl_combined_ctx": "RRL + per-context",
    "rrl_combined_ctx_verified": "\\textbf{RRL full}",
    "oracle": "\\textit{Best possible}",
    "rrl_ce": "Relevance + reputation only",
    "rrl_combined_globalrel": "Global-only source trust",
    "rrl_combined_nocd": "No change detection",
    "rrl_combined_cdold": "Fixed-threshold detector",
    "rrl_combined_e0": "Exploration off",
    "rrl_combined_e10": "Exploration 0.10",
    "rrl_combined_conf": "Confidence-gated exploration",
}
SCEN = {
    "stationary": "Verified",
    "spot_check": "Spot check",
    "update": "Update",
    "update_spot_check": "Update + spot",
    "redesign": "Redesign",
    "redesign_spot_check": "Redesign + spot",
    "sparse_verifier": "Sparse verifier",
    "sycophantic": "Sycophantic",
    "fooled": "Fooled",
    "drift": "Drift",
    "drift_fooled": "Drift + fooled",
    "drift_sycophantic": "Drift + syco.",
}


def load(stem: str, sc: str) -> dict:
    with open(os.path.join(RES, f"{stem}_{sc}.json")) as f:
        return json.load(f)


def pct(x: float) -> str:
    return f"{x * 100:.1f}"


def table(name, caption, label, stem, scens, policies, heldout=True, best_from=None):
    data = {sc: load(stem, sc) for sc in scens}
    cols = "l" + "r" * len(scens)
    head = " & ".join(SCEN[s] for s in scens)
    rows = []
    # bold the best non-oracle overall value per column
    best = {}
    for sc in scens:
        cands = [p for p in (best_from or policies) if p != "oracle" and p in data[sc]["overall"]]
        best[sc] = max(cands, key=lambda p: data[sc]["overall"][p][0])
    for p in policies:
        cells = []
        for sc in scens:
            o = data[sc]["overall"].get(p)
            if o is None:
                cells.append("--")
                continue
            v = pct(o[0])
            if p == best[sc]:
                v = f"\\textbf{{{v}}}"
            h = data[sc].get("heldout", {}).get("pass_rate", {}).get(p) if heldout else None
            cells.append(f"{v} / {pct(h[0])}" if h else v)
        rows.append(f"{LABEL[p]} & " + " & ".join(cells) + " \\\\")
        if p in ("memrl_q", "query_memory_robust") and policies[policies.index(p) + 1 :][:1] == [
            "rrl_combined"
        ]:
            rows.append("\\midrule")
    sep = "\\setlength{\\tabcolsep}{3.5pt}\n" if len(scens) > 4 else ""
    note = "Overall / held-out success (\\%)" if heldout else "Success (\\%)"
    body = "\n".join(rows)
    tex = f"""\\begin{{table*}}[t]
\\centering
\\small
{sep}\\begin{{tabular}}{{{cols}}}
\\toprule
{note} & {head} \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}
\\caption{{{caption}}}
\\label{{{label}}}
\\end{{table*}}
"""
    with open(os.path.join(OUT, name + ".tex"), "w") as f:
        f.write(tex)


def live_table():
    rows = []
    for p in [
        "static_ce",
        "llm_picks",
        "query_memory",
        "query_memory_robust",
        "memrl_q",
        "rrl_combined",
        "rrl_combined_ctx_verified",
    ]:
        cells = []
        for sc in ("stationary", "update"):
            r = load("gate_m_replay", sc)["overall"][p][0]
            lv = load("gate_m_live", sc)["overall"][p][0]
            cells += [pct(r), pct(lv)]
        rows.append(f"{LABEL[p]} & " + " & ".join(cells) + " \\\\")
    tex = (
        """\\begin{table}[t]
\\centering
\\small
\\begin{tabular}{lrrrr}
\\toprule
 & \\multicolumn{2}{c}{Verified} & \\multicolumn{2}{c}{Update} \\\\
Success (\\%) & Replay & Live & Replay & Live \\\\
\\midrule
"""
        + "\n".join(rows)
        + """
\\bottomrule
\\end{tabular}
\\caption{Gate M replayed from recorded MCP responses (30 seeds) versus run with real MCP calls on
every step (3 seeds).}
\\label{tab:live}
\\end{table}
"""
    )
    with open(os.path.join(OUT, "live.tex"), "w") as f:
        f.write(tex)


def numbers():
    """LaTeX macros for numbers quoted in the text."""
    m = {}

    def diff(stem, sc, key):
        c = load(stem, sc)["comparisons"]["overall"][key]
        return c["mean_diff"] * 100, c["ci95"][0] * 100, c["ci95"][1] * 100, c["wins"], c["losses"]

    spot = load("gate_m_replay", "spot_check")
    ver = load("gate_m_replay", "stationary")
    m["MspotRRL"] = pct(spot["overall"]["rrl_combined_ctx_verified"][0])
    m["MspotMemRL"] = pct(spot["overall"]["memrl_q"][0])
    m["MspotRobust"] = pct(spot["overall"]["query_memory_robust"][0])
    m["MverRRL"] = pct(ver["overall"]["rrl_combined_ctx_verified"][0])
    m["MverRobust"] = pct(ver["overall"]["query_memory_robust"][0])
    m["MverLLM"] = pct(ver["overall"]["llm_picks"][0])
    m["MupLLM"] = pct(load("gate_m_replay", "update")["overall"]["llm_picks"][0])
    m["MverPoisonRRL"] = pct(ver["poisoned_choice_rate"]["rrl_combined_ctx_verified"])
    m["MverPoisonLLM"] = pct(ver["poisoned_choice_rate"]["llm_picks"])
    m["MverSendReport"] = pct(ver["poisoned_choice_rate"]["llm_picks_send_report"])
    m["MspotPoisonRRL"] = pct(spot["poisoned_choice_rate"]["rrl_combined_ctx_verified"])
    m["MspotPoisonMemRL"] = pct(spot["poisoned_choice_rate"]["memrl_q"])
    m["MspotPoisonRobust"] = pct(spot["poisoned_choice_rate"]["query_memory_robust"])
    m["MupSendReport"] = pct(
        load("gate_m_replay", "update")["poisoned_choice_rate"]["llm_picks_send_report"]
    )
    d = diff("gate_m_replay", "spot_check", "rrl_combined_ctx_verified_vs_query_memory_robust")
    m["MspotVsRobust"] = f"{d[0]:+.1f}"
    m["MspotVsRobustCI"] = f"[{d[1]:+.1f}, {d[2]:+.1f}]"
    m["MspotVsRobustWL"] = f"{d[3]} of {d[3] + d[4]}"
    d = diff("gate_m_replay", "stationary", "rrl_combined_ctx_verified_vs_query_memory_robust")
    m["MverVsRobust"] = f"{d[0]:+.1f}"
    m["MverVsRobustCI"] = f"[{d[1]:+.1f}, {d[2]:+.1f}]"
    m["MverRobustLead"] = f"{-d[0]:.1f}"
    m["MverRobustLeadCI"] = f"[{-d[2]:.1f}, {-d[1]:.1f}]"
    d = diff("gate_w", "spot_check", "rrl_combined_ctx_verified_vs_query_memory_robust")
    m["WspotVsRobust"] = f"{d[0]:+.1f}"
    d = diff("gate_w", "spot_check", "rrl_combined_globalrel_vs_rrl_combined")
    m["WspotGlobalRelLoss"] = f"{-d[0]:.1f}"
    d = diff("gate_m_replay", "spot_check", "rrl_combined_globalrel_vs_rrl_combined")
    m["MspotGlobalRelLoss"] = f"{-d[0]:.1f}"
    d = diff("gate_m_replay", "stationary", "rrl_combined_ctx_verified_vs_rrl_combined_ctx")
    m["MverVerifiedGain"] = f"{d[0]:+.1f}"
    d = diff("gate_g_hard", "drift", "rrl_combined_ctx_verified_vs_memrl_q")
    m["GdriftVsMemRL"] = f"{d[0]:+.1f}"
    m["GdriftMemRLLead"] = f"{-d[0]:.1f}"
    f_st = load("gate_f_leakfree", "stationary_gemini-2.5-flash")["overall"]
    m["FnoRet"] = pct(f_st["no_retrieval"][0])
    m["Fstatic"] = pct(f_st["static_ce"][0])
    m["Foracle"] = pct(f_st["oracle"][0])
    resets = load("gate_w", "stationary")["change_detector_resets"]
    m["WresetsOld"] = f"{resets['rrl_combined_cdold'][0]:.1f}"
    m["WresetsNew"] = f"{resets['rrl_combined'][0]:.1f}"
    with open(os.path.join(OUT, "numbers.tex"), "w") as f:
        for k, v in m.items():
            f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")


def main():
    os.makedirs(OUT, exist_ok=True)
    main_pol = [
        "static_ce",
        "query_memory",
        "query_memory_robust",
        "memrl_q",
        "rrl_combined",
        "rrl_combined_ctx_verified",
        "oracle",
    ]
    mcp_pol = [
        "static_ce",
        "llm_picks",
        "query_memory",
        "query_memory_robust",
        "memrl_q",
        "rrl_combined",
        "rrl_combined_ctx_verified",
        "oracle",
    ]
    table(
        "gate_m",
        "Gate M: tool selection over live MCP servers with a poisoned server "
        "(replay of recorded MCP responses, 30 seeds). Best non-oracle result per column in bold.",
        "tab:gatem",
        "gate_m_replay",
        ["stationary", "spot_check", "update", "update_spot_check"],
        mcp_pol,
    )
    table(
        "gate_m_clean",
        "Gate M without the poisoned server (clean registry).",
        "tab:gatemclean",
        "gate_m_replay_clean",
        ["stationary", "spot_check", "update", "update_spot_check"],
        mcp_pol,
    )
    table(
        "gate_w",
        "Gate W: web-scraping extractor selection with executable extractors.",
        "tab:gatew",
        "gate_w",
        ["stationary", "spot_check", "redesign", "redesign_spot_check"],
        main_pol,
    )
    table(
        "gate_t_toolbench",
        "Gate T (ToolBench requests from ToolRet): tool selection on real tool descriptions.",
        "tab:gatet",
        "gate_t_toolbench",
        ["stationary", "sycophantic", "fooled", "drift", "drift_fooled"],
        main_pol,
    )
    table(
        "gate_t_toollens",
        "Gate T (ToolLens requests from ToolRet).",
        "tab:gatetl",
        "gate_t_toollens",
        ["stationary", "sycophantic", "fooled", "drift", "drift_fooled"],
        main_pol,
    )
    table(
        "gate_g",
        "Gate G (hard): code generation against a generated library with poisoned "
        "guides and a mid-stream API change.",
        "tab:gateg",
        "gate_g_hard",
        ["stationary", "fooled", "drift", "drift_sycophantic", "drift_fooled"],
        main_pol,
    )
    abl = [
        "rrl_ce",
        "rrl_combined_globalrel",
        "rrl_combined_nocd",
        "rrl_combined_cdold",
        "rrl_combined_e0",
        "rrl_combined_e10",
        "rrl_combined_conf",
        "rrl_combined",
        "rrl_combined_ctx",
        "rrl_combined_ctx_verified",
    ]
    table(
        "ablation_m",
        "Ablations on Gate M (overall / held-out success, \\%). Each row changes one "
        "component of RRL (global); the last two rows add per-context counters and verified memory.",
        "tab:ablation",
        "gate_m_replay",
        ["stationary", "spot_check", "update", "update_spot_check"],
        abl,
        best_from=abl,
    )
    live_table()
    numbers()
    print("tables written to", OUT)


if __name__ == "__main__":
    main()
