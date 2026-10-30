"""
Generate the paper's LaTeX tables and number macros from sim/results/*.json, so no number in the
paper is typed by hand.

    python3 paper/make_tables.py      # writes paper/tables/*.tex and paper/tables/numbers.tex
"""

import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.abspath(os.path.join(HERE, "..", "sim", "results"))
OUT = os.path.join(HERE, "tables")

LABEL = {
    "static_ce": "Static reranker",
    "llm_picks": "LLM picks from tool list",
    "query_memory": "Per-context memory",
    "query_memory_robust": "Per-context memory, robust",
    "memrl_q": "MemRL-style",
    "memrl_official": "MemRL (official rule)",
    "rated": "RATED",
    "rated_verified": "RATED, verified only",
    "cesa": "ESA (CESA-LinUCB)",
    "rrl_combined": "RRL (global)",
    "rrl_combined_ctx": "RRL + per-context",
    "rrl_combined_ctx_verified": "\\textbf{RRL full}",
    "rrl_combined_ctx_verified_fixed": "RRL full, fixed rule",
    "oracle": "\\textit{Best possible}",
    "rrl_ce": "No learned trust, no change detection",
    "rrl_combined_globalrel": "Global-only source trust",
    "rrl_combined_nocd": "No change detection",
    "rrl_combined_cdold": "Fixed-threshold detector, soft outcomes",
    "rrl_combined_cdsoft": "z-test on soft outcomes",
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
    "defect": "Defection",
    "defect_spot_check": "Defection + spot",
}


def load(stem: str, sc: str) -> dict:
    with open(os.path.join(RES, f"{stem}_{sc}.json")) as f:
        return json.load(f)


def pct(x: float) -> str:
    return f"{x * 100:.1f}"


def table(name, caption, label, stem, scens, policies, heldout=True, best_from=None, env="table*"):
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
        prev = policies[policies.index(p) - 1] if policies.index(p) else None
        if p.startswith("rrl") and prev is not None and not prev.startswith("rrl"):
            rows.append("\\midrule")
        rows.append(f"{LABEL[p]} & " + " & ".join(cells) + " \\\\")
    sep = "\\setlength{\\tabcolsep}{3.5pt}\n" if len(scens) > 4 else ""
    note = "Overall / held-out success (\\%)" if heldout else "Success (\\%)"
    body = "\n".join(rows)
    place = "[H]" if env == "table" else "[t]"
    tex = f"""\\begin{{{env}}}{place}
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
\\end{{{env}}}
"""
    with open(os.path.join(OUT, name + ".tex"), "w") as f:
        f.write(tex)


TRUST_STEMS = [
    ("gate_m_replay", "M"),
    ("gate_m_replay_clean", "M clean"),
    ("gate_w", "W"),
    ("gate_t_toolbench", "T ToolBench"),
    ("gate_t_toollens", "T ToolLens"),
    ("gate_g_hard", "G"),
    ("gate_x_mcptox", "X (MCPTox)"),
    ("gate_v_mcpevol", "V (MCPEvol)"),
]
EXTERNAL = ["memrl_q", "memrl_official", "rated", "rated_verified", "cesa"]
# Scenarios for the cross-gate comparison with the external baselines.
EXT_CELLS = [
    ("gate_m_replay", "stationary", "M", "Verified"),
    ("gate_m_replay", "spot_check", "M", "Spot"),
    ("gate_m_replay", "defect_spot_check", "M", "Def.+spot"),
    ("gate_w", "spot_check", "W", "Spot"),
    ("gate_t_toolbench", "fooled", "T-TB", "Fooled"),
    ("gate_t_toollens", "fooled", "T-TL", "Fooled"),
    ("gate_g_hard", "drift", "G", "Drift"),
    ("gate_x_mcptox", "stationary", "X", "Verified"),
    ("gate_x_mcptox", "spot_check", "X", "Spot"),
    ("gate_v_mcpevol", "stationary", "V", "Verified"),
    ("gate_v_mcpevol", "spot_check", "V", "Spot"),
]
LEARNED = "rrl_combined_ctx_verified_vs_rrl_combined_ctx_verified_fixed"
GLOBAL = "rrl_combined_globalrel_vs_rrl_combined"


def holm_reject(ps, alpha=0.05):
    """Holm step-down: which hypotheses are rejected at family-wise level alpha."""
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    out = [False] * len(ps)
    for rank, i in enumerate(order):
        if ps[i] > alpha / (len(ps) - rank):
            break
        out[i] = True
    return out


def trust_rows():
    """Every scenario with unverified feedback: learned trust vs the fixed rule (RRL full), and
    global-only vs per-tool trust (RRL global). Fully verified scenarios are skipped: there the
    learning target is the verifier's result and the variants coincide."""
    rows = []
    for stem, gate in TRUST_STEMS:
        for f in sorted(os.listdir(RES)):
            if not (f.startswith(stem + "_") and f.endswith(".json")):
                continue
            sc = f[len(stem) + 1 : -5]
            if stem == "gate_m_replay" and sc.startswith("clean_"):
                continue
            d = load(stem, sc)
            ps = d["per_seed_overall"]
            if ps["rrl_combined_ctx_verified"] == ps["rrl_combined_ctx_verified_fixed"]:
                continue
            rows.append((gate, sc, d["comparisons"]["overall"]))
    return rows


def minus(v: str) -> str:
    """Typeset a leading or bracketed minus sign as a real minus, leaving en dashes (--) alone."""
    return re.sub(r"(?<![-\d])-(?=\d)", "$-$", v)


def fmt_diff(c):
    lo, hi = c["ci95"]
    v = minus(f"{c['mean_diff'] * 100:+.1f}")
    v = f"\\textbf{{{v}}}" if lo > 0 or hi < 0 else v
    return f"{v} [{minus(f'{lo * 100:+.1f}')}, {minus(f'{hi * 100:+.1f}')}]"


def trust_table():
    body = []
    for gate, sc, c in trust_rows():
        name = SCEN.get(sc, sc.replace("_", " "))
        body.append(f"{gate} & {name} & {fmt_diff(c[LEARNED])} & {fmt_diff(c[GLOBAL])} \\\\")
    tex = (
        "\\begin{table}[H]\n\\centering\n\\small\n\\begin{tabular}{llrr}\n\\toprule\n"
        "Gate & Scenario & Learned $-$ fixed & Global-only $-$ per-tool \\\\\n\\midrule\n"
        + "\n".join(body)
        + "\n\\bottomrule\n\\end{tabular}\n"
        "\\caption{Learned source trust against the fixed discount (RRL full), and global-only "
        "against per-tool trust (RRL global), in every scenario with unverified feedback. Mean paired "
        "difference in overall success (points) over 30 seeds with 95\\% CI; bold where the CI "
        "excludes zero.}\n\\label{tab:trust}\n\\end{table}\n"
    )
    with open(os.path.join(OUT, "trust.tex"), "w") as f:
        f.write(tex)


SHORT = {
    "memrl_q": "MemRL-style",
    "memrl_official": "MemRL (official)",
    "rated": "RATED",
    "rated_verified": "RATED (verified)",
    "cesa": "ESA",
}


def external_table():
    """RRL full against the external baselines: paired difference (points), 30 seeds."""
    head = " & ".join(SHORT[b] for b in EXTERNAL)
    body = []
    for stem, sc, g, scn in EXT_CELLS:
        cells = []
        for b in EXTERNAL:
            c = load(stem, sc)["comparisons"]["overall"][f"rrl_combined_ctx_verified_vs_{b}"]
            lo, hi = c["ci95"]
            v = minus(f"{c['mean_diff'] * 100:+.1f}")
            cells.append(f"\\textbf{{{v}}}" if lo > 0 or hi < 0 else v)
        body.append(f"{g} & {scn} & " + " & ".join(cells) + " \\\\")
    tex = (
        "\\begin{table}[t]\n\\centering\n\\small\n\\setlength{\\tabcolsep}{4pt}\n"
        "\\begin{tabular}{ll" + "r" * len(EXTERNAL) + "}\n\\toprule\n"
        f"Gate & Scenario & {head} \\\\\n\\midrule\n"
        + "\n".join(body)
        + "\n\\bottomrule\n\\end{tabular}\n"
        "\\caption{RRL full minus each of the closest methods, in the same pools with the same "
        "feedback: mean paired difference in overall success (points) over 30 seeds; bold where the "
        "95\\% CI excludes zero; positive favours RRL. Adaptations of each method are in "
        "Section~\\ref{sec:bench}.}\n"
        "\\label{tab:external}\n\\end{table}\n"
    )
    with open(os.path.join(OUT, "external.tex"), "w") as f:
        f.write(tex)


def mcptox_models_table():
    """MCPTox's own per-agent rates on the kept requests (from its labels, no selection)."""
    models = load("gate_x_mcptox", "stationary")["models"]
    rows = []
    for name, r in sorted(models.items(), key=lambda kv: -kv[1]["task_done"]):
        cells = [pct(r["task_done"]), pct(r["attack_success_of_valid"]), pct(r["direct_execution"])]
        label = name.replace("_", "\\_")
        rows.append(f"{label} & {r['n']} & " + " & ".join(cells) + " \\\\")
    tex = (
        "\\begin{table}[H]\n\\centering\n\\small\n\\begin{tabular}{lrrrr}\n\\toprule\n"
        "Agent & Requests & Task done & Attack success & Poisoned tool called \\\\\n"
        " &  & (\\%) & (\\% of valid) & (\\%) \\\\\n\\midrule\n"
        + "\n".join(rows)
        + "\n\\bottomrule\n\\end{tabular}\n"
        "\\caption{MCPTox's recorded agents on the requests Gate X keeps, from MCPTox's own labels: "
        "the share that did exactly what the user asked, attacks that succeeded among valid "
        "outputs, and calls to the poisoned tool itself.}\n\\label{tab:mcptoxmodels}\n\\end{table}\n"
    )
    with open(os.path.join(OUT, "mcptox_models.tex"), "w") as f:
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
    m["WresetsSoft"] = f"{resets['rrl_combined_cdsoft'][0]:.1f}"
    mres = load("gate_m_replay", "stationary")["change_detector_resets"]
    m["MresetsNew"] = f"{mres['rrl_combined'][0]:.1f}"
    m["MresetsSoft"] = f"{mres['rrl_combined_cdsoft'][0]:.1f}"
    fixed = "rrl_combined_ctx_verified_fixed"
    m["MspotFixed"] = pct(spot["overall"][fixed][0])
    d = diff("gate_m_replay", "spot_check", f"rrl_combined_ctx_verified_vs_{fixed}")
    m["MspotFullVsFixed"] = f"{d[0]:+.1f}"
    m["MspotFullVsFixedCI"] = f"[{d[1]:+.1f}, {d[2]:+.1f}]"
    m["MspotFixedLead"] = f"{-d[0]:.1f}"
    m["MspotFixedLeadCI"] = f"[{-d[2]:.1f}, {-d[1]:.1f}]"
    m["MspotCtxGain"] = pct(
        spot["overall"]["rrl_combined_ctx_verified"][0] - spot["overall"]["rrl_combined"][0]
    )
    m["MverCtxGain"] = pct(
        ver["overall"]["rrl_combined_ctx_verified"][0] - ver["overall"]["rrl_combined"][0]
    )
    for stem, key in (("gate_m_replay", "McdCost"), ("gate_w", "WcdCost")):
        o = load(stem, "stationary")["overall"]
        m[key] = pct(o["rrl_combined_nocd"][0] - o["rrl_combined"][0])
    dv = load("gate_m_replay", "defect")["overall"]
    ds = load("gate_m_replay", "defect_spot_check")["overall"]
    m["MdefRRL"] = pct(dv["rrl_combined_ctx_verified"][0])
    m["MdefRobust"] = pct(dv["query_memory_robust"][0])
    m["MdefOracle"] = pct(dv["oracle"][0])
    m["MdefSpotRRL"] = pct(ds["rrl_combined_ctx_verified"][0])
    m["MdefSpotMemory"] = pct(ds["query_memory"][0])
    m["MdefSpotFixed"] = pct(ds[fixed][0])
    m["MdefSpotRobust"] = pct(ds["query_memory_robust"][0])
    m["MdefSpotDrop"] = pct(
        spot["overall"]["rrl_combined_ctx_verified"][0] - ds["rrl_combined_ctx_verified"][0]
    )
    m["MdefOracleDrop"] = pct(spot["overall"]["oracle"][0] - ds["oracle"][0])
    m["MdefSpotNocd"] = pct(ds["rrl_combined_nocd"][0])
    m["MdefSpotCd"] = pct(ds["rrl_combined"][0])
    d = diff(
        "gate_m_replay", "defect_spot_check", "rrl_combined_ctx_verified_vs_query_memory_robust"
    )
    m["MdefSpotVsRobust"] = f"{d[0]:+.1f}"
    cs = load("gate_m_replay_clean", "defect_spot_check")["overall"]
    m["McleanDefSpotRRL"] = pct(cs["rrl_combined_ctx_verified"][0])
    m["McleanDefSpotRobust"] = pct(cs["query_memory_robust"][0])
    tr = [c[LEARNED] for _, _, c in trust_rows()]
    m["TrustN"] = str(len(tr))
    m["TrustBetter"] = str(sum(c["ci95"][0] > 0 for c in tr))
    m["TrustWorse"] = str(sum(c["ci95"][1] < 0 for c in tr))
    m["TrustMin"] = f"{min(c['mean_diff'] for c in tr) * 100:+.1f}"
    m["TrustMax"] = f"{max(c['mean_diff'] for c in tr) * 100:+.1f}"
    holm = holm_reject([c["t_test_p"] for c in tr])
    m["TrustHolmBetter"] = str(sum(r and c["mean_diff"] > 0 for r, c in zip(holm, tr)))
    m["TrustHolmWorse"] = str(sum(r and c["mean_diff"] < 0 for r, c in zip(holm, tr)))
    m["TrustMaxAbs"] = f"{max(abs(c['mean_diff']) for c in tr) * 100:.1f}"

    def ov(stem, sc, pol):
        return load(stem, sc)["overall"][pol][0] * 100

    def ho(stem, sc, pol):
        return load(stem, sc)["heldout"]["pass_rate"][pol][0] * 100

    def span(xs):
        return f"{min(xs):.1f}--{max(xs):.1f}"

    full, robust, mem = "rrl_combined_ctx_verified", "query_memory_robust", "query_memory"
    m["MspotMemory"] = f"{ov('gate_m_replay', 'spot_check', mem):.1f}"
    for key, pol in (("RRL", full), ("Fixed", fixed), ("Memory", mem), ("Robust", robust)):
        m[f"MspotHeld{key}"] = f"{ho('gate_m_replay', 'spot_check', pol):.1f}"
    m["MspotPoisonFixed"] = pct(spot["poisoned_choice_rate"][fixed])
    poisoned_spot = [
        ("gate_m_replay", "spot_check"),
        ("gate_m_replay", "update_spot_check"),
        ("gate_m_replay", "defect_spot_check"),
        ("gate_w", "spot_check"),
        ("gate_w", "redesign_spot_check"),
    ]
    m["TrustPoisonRange"] = span([ov(st, sc, fixed) - ov(st, sc, full) for st, sc in poisoned_spot])
    m["TrustSame"] = str(len(tr) - int(m["TrustBetter"]) - int(m["TrustWorse"]))
    m["SpotVsRobustRange"] = span(
        [ov(st, sc, full) - ov(st, sc, robust) for st, sc in poisoned_spot[:2] + poisoned_spot[3:]]
    )
    m["WverLead"] = span(
        [ov("gate_w", sc, robust) - ov("gate_w", sc, full) for sc in ("stationary", "redesign")]
    )
    m["CleanVerLead"] = span(
        [
            ov("gate_m_replay_clean", sc, robust) - ov("gate_m_replay_clean", sc, full)
            for sc in ("stationary", "update")
        ]
    )
    m["CleanSpotLead"] = span(
        [
            ov("gate_m_replay_clean", sc, robust) - ov("gate_m_replay_clean", sc, full)
            for sc in ("spot_check", "update_spot_check")
        ]
    )
    m["GdriftGlobalLag"] = (
        f"{ov('gate_g_hard', 'drift', 'memrl_q') - ov('gate_g_hard', 'drift', 'rrl_combined'):.1f}"
    )
    m["MdefSpotResets"] = (
        f"{load('gate_m_replay', 'defect_spot_check')['change_detector_resets'][full][0]:.1f}"
    )
    # verified-outcome detector vs the soft-outcome z-test (RRL global), every scenario
    soft_better = soft_worse = n_cd = 0
    for f_ in sorted(os.listdir(RES)):
        if not (f_.startswith("gate_") and f_.endswith(".json")):
            continue
        with open(os.path.join(RES, f_)) as fh:
            c = (
                json.load(fh)
                .get("comparisons", {})
                .get("overall", {})
                .get("rrl_combined_cdsoft_vs_rrl_combined")
            )
        if c is None:
            continue
        n_cd += 1
        soft_better += c["ci95"][0] > 0
        soft_worse += c["ci95"][1] < 0
    m["CdN"], m["CdVerBetter"], m["CdSoftBetter"] = str(n_cd), str(soft_worse), str(soft_better)
    # Gate T: RRL full against MemRL-style and per-context memory
    t_cells = [
        (st, sc)
        for st in ("gate_t_toolbench", "gate_t_toollens")
        for sc in (
            "stationary",
            "sparse_verifier",
            "sycophantic",
            "fooled",
            "drift",
            "drift_fooled",
        )
    ]
    m["TsycoMemrlLead"] = span(
        [
            ov(st, "sycophantic", "memrl_q") - ov(st, "sycophantic", full)
            for st in ("gate_t_toolbench", "gate_t_toollens")
        ]
    )
    m["TBsparseMemrlLead"] = (
        f"{ov('gate_t_toolbench', 'sparse_verifier', 'memrl_q') - ov('gate_t_toolbench', 'sparse_verifier', full):.1f}"
    )
    m["TLrrlLead"] = span(
        [
            ov("gate_t_toollens", sc, full) - ov("gate_t_toollens", sc, "memrl_q")
            for sc in ("stationary", "fooled")
        ]
    )
    m["TqmRange"] = span(
        [ov(st, sc, p_) - ov(st, sc, mem) for st, sc in t_cells for p_ in (full, "memrl_q")]
    )
    m["TeTenWins"] = str(
        sum(ov(st, sc, "rrl_combined_e10") >= ov(st, sc, "memrl_q") - 0.5 for st, sc in t_cells)
    )
    m["TcellsN"] = str(len(t_cells))
    rated_minus_rrl = [ov(st, sc, "rated") - ov(st, sc, full) for st, sc in t_cells]
    m["TRatedAhead"] = f"{max(rated_minus_rrl):.1f}"
    m["TRatedBehind"] = f"{-min(rated_minus_rrl):.1f}"
    e_cells = [
        (st, sc)
        for st in ("gate_t_toolbench", "gate_t_toollens", "gate_g_hard")
        for sc in (
            "stationary",
            "sparse_verifier",
            "sycophantic",
            "fooled",
            "drift",
            "drift_fooled",
            "drift_sycophantic",
        )
    ]
    e_cmp = [
        load(st, sc)["comparisons"]["overall"]["rrl_combined_e10_vs_rrl_combined"]
        for st, sc in e_cells
    ]
    m["ETenBetter"] = str(sum(c["ci95"][0] > 0 for c in e_cmp))
    m["ETenWorse"] = str(sum(c["ci95"][1] < 0 for c in e_cmp))
    m["ETenN"] = str(len(e_cmp))

    # The closest methods (Section "Against the closest methods").
    def cmp_(stem, sc, b):
        c = load(stem, sc)["comparisons"]["overall"][f"rrl_combined_ctx_verified_vs_{b}"]
        return c["mean_diff"] * 100, c["ci95"][0] * 100, c["ci95"][1] * 100

    for key, pol in (
        ("Rated", "rated"),
        ("RatedVer", "rated_verified"),
        ("MemOff", "memrl_official"),
        ("Cesa", "cesa"),
    ):
        m[f"Mspot{key}"] = pct(spot["overall"][pol][0])
        m[f"Mspot{key}Poison"] = pct(spot["poisoned_choice_rate"][pol])
        m[f"Mver{key}"] = pct(ver["overall"][pol][0])
    d = cmp_("gate_m_replay", "spot_check", "rated_verified")
    m["MspotVsRatedVer"] = f"{d[0]:+.1f}"
    m["MspotVsRatedVerCI"] = f"[{d[1]:+.1f}, {d[2]:+.1f}]"
    d = cmp_("gate_m_replay", "update_spot_check", "rated_verified")
    m["MupspotVsRatedVer"] = f"{d[0]:+.1f}"
    m["MupspotVsRatedVerCI"] = f"[{d[1]:+.1f}, {d[2]:+.1f}]"
    m["MupspotVsRatedVerAbs"] = f"{d[0]:.1f}"
    m["WspotRatedVerLead"] = span(
        [-cmp_("gate_w", sc, "rated_verified")[0] for sc in ("spot_check", "redesign_spot_check")]
    )
    d = cmp_("gate_m_replay", "stationary", "rated")
    m["MverVsRated"] = f"{d[0]:+.1f}"
    m["MverVsRatedCI"] = f"[{d[1]:+.1f}, {d[2]:+.1f}]"
    m["MverVsRatedAbs"] = f"{d[0]:.1f}"
    m["CleanVerRated"] = f"{ov('gate_m_replay_clean', 'stationary', 'rated'):.1f}"
    m["CleanVerRRL"] = f"{ov('gate_m_replay_clean', 'stationary', full):.1f}"
    faced = [
        ov(st, sc, p_)
        for st, sc in poisoned_spot
        for p_ in ("rated", "memrl_official", "memrl_q", "cesa")
    ]
    m["FaceValueMax"] = f"{max(faced):.1f}"
    # Gate X (MCPTox) and Gate V (MCPEvol-Bench).
    xv, xs = load("gate_x_mcptox", "stationary"), load("gate_x_mcptox", "spot_check")
    for key, pol in (
        ("Memory", mem),
        ("Rated", "rated"),
        ("RRL", full),
        ("Emb", "static_rrf"),
        ("CE", "static_ce"),
        ("LLM", "llm_picks"),
    ):
        m[f"Xver{key}"] = pct(xv["overall"][pol][0])
        m[f"Xspot{key}"] = pct(xs["overall"][pol][0])
    m["XverPoisonRRL"] = pct(xv["poisoned_choice_rate"][full])
    m["XspotPoisonRRL"] = pct(xs["poisoned_choice_rate"][full])
    m["XpoisonCE"] = pct(xv["poisoned_choice_rate"]["static_ce"])
    m["XattackLLM"] = pct(xv["attack_choice_rate"]["llm_picks"])
    sel = [v for k, v in xs["attack_choice_rate"].items() if k != "llm_picks"]
    sel += [v for k, v in xv["attack_choice_rate"].items() if k != "llm_picks"]
    m["XattackSelMax"] = pct(max(sel))
    d = cmp_("gate_x_mcptox", "stationary", mem)
    m["XverMemoryLead"] = f"{-d[0]:.1f}"
    d = cmp_("gate_x_mcptox", "spot_check", "rated")
    m["XspotRatedLead"] = f"{-d[0]:.1f}"
    m["XspotRatedLeadCI"] = f"[{-d[2]:.1f}, {-d[1]:.1f}]"
    vv, vs = load("gate_v_mcpevol", "stationary"), load("gate_v_mcpevol", "spot_check")
    for key, pol in (("RRL", full), ("Rated", "rated"), ("MemRL", "memrl_q"), ("Memory", mem)):
        m[f"Vver{key}"] = pct(vv["overall"][pol][0])
        m[f"Vspot{key}"] = pct(vs["overall"][pol][0])
    d = cmp_("gate_v_mcpevol", "stationary", "rated")
    m["VverVsRated"] = f"{d[0]:+.1f}"
    m["VverVsRatedCI"] = f"[{d[1]:+.1f}, {d[2]:+.1f}]"
    d = cmp_("gate_v_mcpevol", "spot_check", "rated")
    m["VspotRatedLead"] = f"{-d[0]:.1f}"
    m["VspotRatedLeadCI"] = f"[{-d[2]:.1f}, {-d[1]:.1f}]"
    data_dir = os.path.abspath(os.path.join(HERE, "..", "data"))
    with open(os.path.join(data_dir, "gate_x", "meta.json")) as fh:
        xm = json.load(fh)
    with open(os.path.join(data_dir, "gate_v", "meta.json")) as fh:
        vm = json.load(fh)
    m["XInst"] = f"{xm['instances']:,}"
    m["XCtx"] = str(xm["contexts"])
    m["XDropped"] = str(sum(xm["dropped"].values()))
    m["VReq"] = str(vm["requests"])
    m["VTasks"] = str(vm["tasks"])
    m["VChanged"] = str(vm["changed_pools"])
    with open(os.path.join(OUT, "numbers.tex"), "w") as f:
        for k, v in m.items():
            f.write(f"\\newcommand{{\\{k}}}{{{minus(v)}}}\n")


def main():
    os.makedirs(OUT, exist_ok=True)
    main_pol = [
        "static_ce",
        "query_memory",
        "query_memory_robust",
        "memrl_q",
        *EXTERNAL[1:],
        "rrl_combined",
        "rrl_combined_ctx_verified",
        "rrl_combined_ctx_verified_fixed",
        "oracle",
    ]
    mcp_pol = [
        "static_ce",
        "llm_picks",
        "query_memory",
        "query_memory_robust",
        "memrl_q",
        *EXTERNAL[1:],
        "rrl_combined",
        "rrl_combined_ctx_verified",
        "rrl_combined_ctx_verified_fixed",
        "oracle",
    ]
    table(
        "gate_x",
        "Gate X: tool selection on MCPTox's requests, toolsets and poisoned tools; the correct "
        "tool is the one MCPTox's agents called when they did exactly what the user asked. "
        "LLM picks are MCPTox's recorded Gemini 2.5 Flash responses, scored by its labels. Held-out: "
        "nine servers seen only in a final block.",
        "tab:gatex",
        "gate_x_mcptox",
        ["stationary", "spot_check"],
        mcp_pol,
        env="table",
    )
    table(
        "gate_v",
        "Gate V: tool selection on MCPEvol-Bench's evolving toolsets; the correct tools are those "
        "its successful reference runs called. Held-out: 20\\% of tasks, seen only on the late "
        "toolset.",
        "tab:gatev",
        "gate_v_mcpevol",
        ["stationary", "spot_check"],
        main_pol,
        env="table",
    )
    external_table()
    mcptox_models_table()
    table(
        "gate_m",
        "Gate M: tool selection over local MCP servers with a poisoned server "
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
        "rrl_combined_cdsoft",
        "rrl_combined_e0",
        "rrl_combined_e10",
        "rrl_combined_conf",
        "rrl_combined",
        "rrl_combined_ctx",
        "rrl_combined_ctx_verified",
        "rrl_combined_ctx_verified_fixed",
    ]
    table(
        "ablation_m",
        "Ablations on Gate M (overall / held-out success, \\%). Each row changes one "
        "component of RRL (global); the last three rows add per-context counters, then verified memory, then swap learned trust for the fixed rule.",
        "tab:ablation",
        "gate_m_replay",
        ["stationary", "spot_check", "update", "update_spot_check"],
        abl,
        best_from=abl,
    )
    trust_table()
    table(
        "gate_m_defect",
        "Staged defection on Gate M: halfway through, the most useful honest extractor starts "
        "returning well-formed wrong results while still reporting success. Held-out shops come "
        "after the defection.",
        "tab:defect",
        "gate_m_replay",
        ["defect", "defect_spot_check"],
        mcp_pol,
    )
    table(
        "gate_m_clean_defect",
        "Staged defection on Gate M without the poisoned server.",
        "tab:defectclean",
        "gate_m_replay_clean",
        ["defect", "defect_spot_check"],
        mcp_pol,
        env="table",
    )
    live_table()
    numbers()
    print("tables written to", OUT)


if __name__ == "__main__":
    main()
