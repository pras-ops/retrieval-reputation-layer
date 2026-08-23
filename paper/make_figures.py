"""Generate publication figures for the RRL boundary-study paper.

Reads the committed Gate R results JSONs and the generation cache; writes
vector PDFs into paper/figures/. Run from the repo root:

    python3 paper/make_figures.py
"""

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(OUT, exist_ok=True)

# Okabe-Ito, colorblind-safe (validated: CVD deltaE >= 51 adjacent pairs)
BLUE = "#0072B2"  # hero: gamma=1.0, explore, 16 epochs
ORANGE = "#E69F00"  # gamma=0.95 (decay mistuned)
GREEN = "#009E73"  # gamma=1.0, no exploration
GRAY = "#555555"  # static baseline (reference)

plt.rcParams.update(
    {
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.linestyle": ":",
        "grid.alpha": 0.4,
        "pdf.fonttype": 42,
    }
)


def epoch_means(path: str, epochs: int):
    d = json.load(open(os.path.join(ROOT, path)))
    c = d["rrl_curve"]
    return [sum(c[e * 30 : (e + 1) * 30]) / 30 for e in range(epochs)]


def fig_learning():
    r095 = epoch_means("sim/results/gate_r_real.json", 8)
    rnoex = epoch_means("sim/results/gate_r_gamma1_noexplore.json", 8)
    r16 = epoch_means("sim/results/gate_r_gamma1_e16.json", 16)

    fig, ax = plt.subplots(figsize=(6.0, 3.4))
    x16 = range(1, 17)
    x8 = range(1, 9)

    ax.axhline(0.667, color=GRAY, linewidth=1.5, linestyle="--", zorder=1)
    ax.plot(x16, r16, color=BLUE, linewidth=2.2, marker="o", markersize=3.5, zorder=4)
    ax.plot(x8, r095, color=ORANGE, linewidth=1.6, marker="s", markersize=3, zorder=3)
    ax.plot(x8, rnoex, color=GREEN, linewidth=1.6, marker="^", markersize=3.5, zorder=2)

    # direct labels (relief for contrast; identity never color-alone)
    ax.annotate(
        "RRL  $\\gamma$=1.0, explore (16 ep.)",
        xy=(16, r16[-1]),
        xytext=(9.2, 0.735),
        color=BLUE,
        fontsize=8.5,
        fontweight="bold",
    )
    ax.annotate(
        "RRL  $\\gamma$=0.95 (decay erases evidence)",
        xy=(8, r095[-1]),
        xytext=(4.4, 0.585),
        color=ORANGE,
        fontsize=8.5,
    )
    ax.annotate(
        "RRL  $\\gamma$=1.0, no exploration",
        xy=(8, rnoex[-1]),
        xytext=(1.0, 0.555),
        color=GREEN,
        fontsize=8.5,
    )
    ax.annotate(
        "static cross-encoder baseline (deterministic, 0.667)",
        xy=(1, 0.667),
        xytext=(1.0, 0.673),
        color=GRAY,
        fontsize=8.5,
    )

    ax.set_xlabel("Epoch (each = one pass over the 30 recurring problems)")
    ax.set_ylabel("Unit-test pass rate")
    ax.set_xlim(0.6, 16.4)
    ax.set_ylim(0.54, 0.76)
    ax.set_xticks([1, 2, 4, 6, 8, 10, 12, 14, 16])
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_gate_r_learning.pdf"))
    fig.savefig(os.path.join(OUT, "fig_gate_r_learning.png"), dpi=200)
    plt.close(fig)


def fig_sensitivity():
    cache = os.path.join(ROOT, "data", "mbpp_sweep_cache.jsonl")
    entries = [json.loads(line) for line in open(cache) if line.strip()]
    real = [e for e in entries if e.get("generator") == "gemini-2.5-flash"]
    good = [e for e in real if e["retrieved_id"].startswith("good_")]
    dist = [e for e in real if e["retrieved_id"].startswith("distractor_")]
    pg = sum(e["passed"] for e in good) / len(good)
    pd = sum(e["passed"] for e in dist) / len(dist)

    fig, ax = plt.subplots(figsize=(3.4, 3.0))
    bars = ax.bar(
        [0, 1],
        [pg, pd],
        width=0.55,
        color=[BLUE, GRAY],
        edgecolor="white",
        linewidth=2,
        zorder=3,
    )
    for rect, v, n in zip(bars, [pg, pd], [len(good), len(dist)]):
        ax.annotate(
            f"{v:.3f}\n(n={n})",
            xy=(rect.get_x() + rect.get_width() / 2, v),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            fontsize=9,
        )
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["own reference\nsolution retrieved", "similar distractor\nretrieved"])
    ax.set_ylabel("Unit-test pass rate")
    ax.set_ylim(0, 0.75)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_sensitivity.pdf"))
    fig.savefig(os.path.join(OUT, "fig_sensitivity.png"), dpi=200)
    plt.close(fig)
    return pg, pd, len(good), len(dist)


if __name__ == "__main__":
    fig_learning()
    pg, pd, ng, nd = fig_sensitivity()
    print(f"figures written to {OUT}")
    print(
        f"sensitivity: good={pg:.3f} (n={ng})  distractor={pd:.3f} (n={nd})  spread={pg - pd:+.3f}"
    )
