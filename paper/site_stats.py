"""
Site-level uncertainty for the paper's primary comparison (RRL full vs the robust per-context
memory on Gate M), replayed from the committed outcome records.

Seed-level confidence intervals only vary request order, noise and exploration over the same 30
shops. Here each shop is one unit: its success difference is averaged over seeds and visits, and
a bootstrap over shops gives an interval for the mean shop-level difference.

    python3 paper/site_stats.py      # writes paper/tables/site_stats.tex
"""

import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "sim"))
sys.path.insert(0, os.path.join(HERE, ".."))

import run_gate_m as gm  # noqa: E402

A, B = "query_memory_robust", "rrl_combined_ctx_verified"
SEEDS = range(42, 72)
BOOT = 10_000


def site_diffs(env, scenario):
    gm.RUN_POLICIES[:] = [A, B]
    tot, cnt = {}, {}
    for sd in SEEDS:
        pairs = gm.stream_for(sd, 4, gm.SCENARIOS[scenario])[0]
        hist, *_ = gm.replay_seed(env, sd, 4, gm.SCENARIOS[scenario])
        for i, (site, _) in enumerate(pairs):
            tot[site] = tot.get(site, 0.0) + hist[B][i] - hist[A][i]
            cnt[site] = cnt.get(site, 0) + 1
    return {s: tot[s] / cnt[s] for s in tot}


def boot_ci(xs, rng):
    means = sorted(sum(rng.choice(xs) for _ in xs) / len(xs) for _ in range(BOOT))
    return means[int(0.025 * BOOT)], means[int(0.975 * BOOT)]


def main():
    env = gm.Env(clean=False)
    env.llm = {}
    held = {s["id"] for s in gm.w.sites() if s["heldout"]}
    rng = random.Random(0)
    macros = {}
    for sc, tag in (("spot_check", "Spot"), ("stationary", "Ver")):
        d = site_diffs(env, sc)
        xs = list(d.values())
        lo, hi = boot_ci(xs, rng)
        macros[f"Site{tag}Mean"] = f"{100 * sum(xs) / len(xs):+.1f}"
        macros[f"Site{tag}CI"] = f"[{100 * lo:+.1f}, {100 * hi:+.1f}]"
        macros[f"Site{tag}Wins"] = f"{sum(x > 0 for x in xs)} of {len(xs)}"
        macros[f"Site{tag}Lead"] = f"{-100 * sum(xs) / len(xs):.1f}"
        macros[f"Site{tag}LeadCI"] = f"[{-100 * hi:+.1f}, {-100 * lo:+.1f}]"
        seen = [v for s, v in d.items() if s not in held]
        macros[f"Site{tag}SeenWins"] = f"{sum(x > 0 for x in seen)} of {len(seen)}"
    with open(os.path.join(HERE, "tables", "site_stats.tex"), "w") as f:
        for k, v in macros.items():
            f.write(f"\\newcommand{{\\{k}}}{{{v.replace('[-', '[$-$').replace(' -', ' $-$')}}}\n")
    print(macros)


if __name__ == "__main__":
    main()
