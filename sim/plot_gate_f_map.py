"""
Similarity vs usefulness map from a Gate F outcome table (measured, not illustrative).

For every (query, candidate document) pair in the Gate F pools it plots:
  x  hybrid retrieval similarity (what the vector/BM25 retriever ranks by)
  y  cross-encoder rank score (what a reranker ranks by)
  z  measured usefulness: did the generated code pass the unit tests with this document

and reports how often "similar" and "useful" disagree:
  * pass rate by similarity rank (1 = most similar)
  * queries where the most similar document fails but another candidate passes
  * documents that make the answer worse than retrieving nothing

Seeds whose cells are not all in the table yet are skipped, so it runs on partial tables.

    python3 sim/plot_gate_f_map.py --generator gemini
"""

import argparse
import os
import random
import sys

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from run_gate_matrix import (
    NO_DOC,
    POOL_K,
    RESULTS_DIR,
    OutcomeTable,
    load_or_build_pools,
    needed_cells,
)


def collect(tag: str, seeds, leaky: bool):
    table = OutcomeTable(tag)
    datasets = load_or_build_pools(seeds, leaky, None, None)
    rows, used = [], []
    for s in seeds:
        pools, ce_order = datasets[s]
        if table.missing(needed_cells(pools)):
            continue
        used.append(s)
        for t, pool in pools.items():
            base = table.get(t, NO_DOC)
            order = ce_order[t]
            for rank, (d, sim) in enumerate(pool):
                rows.append(
                    {
                        "seed": s,
                        "task": t,
                        "doc": d,
                        "sim_rank": rank,
                        "sim": sim,
                        "ce": 1.0 - order.index(d) / max(1, len(order) - 1),
                        "passed": table.get(t, d),
                        "lift": table.get(t, d) - base,
                    }
                )
    return rows, used


def report(rows, used) -> dict:
    by_query = {}
    for r in rows:
        by_query.setdefault((r["seed"], r["task"]), []).append(r)
    n_q = len(by_query)
    top_fail_other_pass = sum(
        1
        for rs in by_query.values()
        if min(rs, key=lambda r: r["sim_rank"])["passed"] < 0.5
        and max(r["passed"] for r in rs) > 0.5
    )
    rank_rates = []
    for k in range(POOL_K):
        sel = [r["passed"] for r in rows if r["sim_rank"] == k]
        rank_rates.append(sum(sel) / len(sel) if sel else float("nan"))
    hurt = sum(1 for r in rows if r["lift"] < 0)
    helped = sum(1 for r in rows if r["lift"] > 0)
    out = {
        "seeds_used": len(used),
        "queries": n_q,
        "pairs": len(rows),
        "pass_rate_by_similarity_rank": rank_rates,
        "top_similar_fails_but_another_passes": top_fail_other_pass / n_q if n_q else 0.0,
        "pairs_that_hurt_vs_no_retrieval": hurt / len(rows) if rows else 0.0,
        "pairs_that_help_vs_no_retrieval": helped / len(rows) if rows else 0.0,
    }
    print(f"seeds used: {len(used)}  queries: {n_q}  (query, doc) pairs: {len(rows)}")
    print("pass rate by similarity rank (1 = most similar):")
    for k, v in enumerate(rank_rates):
        print(f"  rank {k + 1}: {v:.3f}")
    print(
        f"most-similar doc fails but another candidate passes: "
        f"{out['top_similar_fails_but_another_passes']:.1%} of queries"
    )
    print(
        f"vs no retrieval: {out['pairs_that_help_vs_no_retrieval']:.1%} of pairs help, "
        f"{out['pairs_that_hurt_vs_no_retrieval']:.1%} hurt"
    )
    return out


def plot(rows, stats, path, title):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rng = random.Random(0)
    jit = lambda: rng.uniform(-0.08, 0.08)  # noqa: E731
    green, red = "#15803d", "#b91c1c"

    fig = plt.figure(figsize=(17, 5.6))

    ax = fig.add_subplot(1, 3, 1, projection="3d")
    for passed, color, label in ((1.0, green, "passed tests"), (0.0, red, "failed tests")):
        sel = [r for r in rows if r["passed"] == passed]
        ax.scatter(
            [r["sim"] + jit() * 0.3 for r in sel],
            [r["ce"] + jit() for r in sel],
            [r["passed"] + jit() for r in sel],
            c=color,
            s=8,
            alpha=0.45,
            label=label,
        )
    ax.set_xlabel("hybrid similarity")
    ax.set_ylabel("cross-encoder rank score")
    ax.set_zlabel("useful? (tests pass)")
    ax.set_title("Every (query, candidate) pair")
    ax.legend(loc="upper left", fontsize=8)

    ax2 = fig.add_subplot(1, 3, 2)
    for passed, color in ((1.0, green), (0.0, red)):
        sel = [r for r in rows if r["passed"] == passed]
        ax2.scatter(
            [r["sim"] + jit() * 0.3 for r in sel],
            [r["lift"] + jit() * 2 for r in sel],
            c=color,
            s=8,
            alpha=0.4,
        )
    ax2.axhline(0, color="#555", linewidth=0.8)
    ax2.set_yticks([-1, 0, 1])
    ax2.set_yticklabels(["hurts\n(fails, but passes\nwith no doc)", "no change", "helps"])
    ax2.set_xlabel("hybrid similarity")
    ax2.set_title("Effect vs retrieving nothing")

    ax3 = fig.add_subplot(1, 3, 3)
    rates = stats["pass_rate_by_similarity_rank"]
    ax3.bar([str(k + 1) for k in range(len(rates))], rates, color="#2563eb")
    ax3.set_ylim(0, 1)
    ax3.set_xlabel("similarity rank in the top-5 pool (1 = most similar)")
    ax3.set_ylabel("unit-test pass rate")
    ax3.set_title(
        f"Most similar fails, another passes: "
        f"{stats['top_similar_fails_but_another_passes']:.0%} of queries"
    )
    for k, v in enumerate(rates):
        ax3.text(k, v + 0.02, f"{v:.2f}", ha="center")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[plot] {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--generator", default="gemini", help="outcome-table tag prefix: gemini or hf")
    ap.add_argument("--hf-model", default="Qwen2.5-0.5B-Instruct")
    ap.add_argument("--seeds", type=int, default=30)
    ap.add_argument("--leaky", action="store_true")
    args = ap.parse_args()

    tag = "gemini-2.5-flash" if args.generator == "gemini" else f"hf-{args.hf_model}"
    rows, used = collect(tag, list(range(42, 42 + args.seeds)), args.leaky)
    if not rows:
        sys.exit("no seed has a complete set of outcome cells yet")
    stats = report(rows, used)
    variant = "leaky" if args.leaky else "leakfree"
    os.makedirs(RESULTS_DIR, exist_ok=True)
    plot(
        rows,
        stats,
        os.path.join(RESULTS_DIR, f"gate_f_map_{variant}_{tag}.png"),
        f"Similar vs useful, measured — MBPP {variant}, {tag}, {len(used)} seeds",
    )


if __name__ == "__main__":
    main()
