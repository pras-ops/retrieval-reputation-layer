"""
Learning curves for the paper (Figure 1), replayed from the committed Gate M outcome records.

    python3 paper/make_figure.py      # writes paper/figures/gate_m_curves.pdf
"""

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "sim"))
sys.path.insert(0, os.path.join(HERE, ".."))

import run_gate_m as gm  # noqa: E402

POLICIES = [
    ("oracle", "Best possible", "#9e9e9e", "--"),
    ("rrl_combined_ctx_verified", "RRL full", "#1b5e9e", "-"),
    ("query_memory_robust", "Per-context memory, robust", "#2e8b57", "-"),
    ("memrl_q", "MemRL-style", "#c0392b", "-"),
    ("llm_picks", "LLM picks from tool list", "#e08e0b", "-"),
    ("static_ce", "Static reranker", "#555555", ":"),
]
SEEDS = range(42, 72)
WINDOW = 25


def curves(env, scenario):
    gm.RUN_POLICIES[:] = [p for p, *_ in POLICIES if p != "llm_picks"]
    total, n_held = None, 0
    for sd in SEEDS:
        hist, _, n_held, _ = gm.replay_seed(env, sd, 4, gm.SCENARIOS[scenario])
        if total is None:
            total = {k: [0.0] * len(v) for k, v in hist.items()}
        for k, v in hist.items():
            total[k] = [a + b / len(SEEDS) for a, b in zip(total[k], v)]
    return total, n_held


def smooth(xs):
    return [
        sum(xs[max(0, i - WINDOW + 1) : i + 1]) / (i + 1 - max(0, i - WINDOW + 1))
        for i in range(len(xs))
    ]


def main():
    env = gm.Env(clean=False)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.9), sharey=True)
    for ax, (sc, title) in zip(
        axes, [("stationary", "Every call verified"), ("spot_check", "20% of calls verified")]
    ):
        total, n_held = curves(env, sc)
        n = len(next(iter(total.values())))
        ax.axvspan(n - n_held, n, color="#eeeeee", zorder=0)
        ax.text(
            n - n_held / 2, 1.02, "held-out", ha="center", va="bottom", fontsize=8, color="#666666"
        )
        for key, label, color, style in POLICIES:
            if key in total:
                ax.plot(
                    smooth(total[key]), label=label, color=color, linestyle=style, linewidth=1.6
                )
        ax.set_title(title, fontsize=10, pad=14)
        ax.set_xlabel("Step")
        ax.set_ylim(-0.02, 1.0)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel(f"Success rate ({WINDOW}-step moving average)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=6, fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    out = os.path.join(HERE, "figures")
    os.makedirs(out, exist_ok=True)
    fig.savefig(os.path.join(out, "gate_m_curves.pdf"))
    print("figure written to", out)


if __name__ == "__main__":
    main()
