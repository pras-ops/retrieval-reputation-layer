"""Export the analysis tables the evaluation plan calls for, as CSV."""
import csv, json, os, collections, statistics as st, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from rrl.metrics import diagnosticity, observations_required

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(ROOT, "data", "analysis")
os.makedirs(OUT, exist_ok=True)


def rows(path):
    p = os.path.join(ROOT, "data", path)
    return [json.loads(l) for l in open(p) if l.strip()] if os.path.exists(p) else []


def own(r):
    return r["retrieved_id"].startswith("good_%d_" % r["task_id"])


# ---- delta_by_task.csv -------------------------------------------------
plus = rows("gemini_cache_mbppplus.jsonl")
graded = rows("gemini_cache_graded.jsonl")
by = collections.defaultdict(lambda: collections.defaultdict(list))
for r in graded:
    by[r["task_id"]]["mbpp_correct" if own(r) else "mbpp_wrong"].append(float(r["passed"]))
for r in plus:
    k = "plus_correct" if own(r) else "plus_wrong"
    by[r["task_id"]][k].append(float(r["mbppplus_fraction"]))

with open(os.path.join(OUT, "delta_by_task.csv"), "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["task_id", "n_correct", "n_wrong", "p_correct_mbpp", "p_wrong_mbpp",
                "delta_mbpp", "p_correct_plus", "p_wrong_plus", "delta_plus", "bucket"])
    for tid in sorted(by):
        d = by[tid]
        if not d["mbpp_correct"] or not d["mbpp_wrong"]:
            continue
        pc, pw = st.mean(d["mbpp_correct"]), st.mean(d["mbpp_wrong"])
        dm = pc - pw
        pcp = st.mean(d["plus_correct"]) if d["plus_correct"] else ""
        pwp = st.mean(d["plus_wrong"]) if d["plus_wrong"] else ""
        dp = (pcp - pwp) if pcp != "" and pwp != "" else ""
        bucket = "<=0" if dm <= 0 else "low" if dm < 0.34 else "medium" if dm < 0.67 else "high"
        w.writerow([tid, len(d["mbpp_correct"]), len(d["mbpp_wrong"]),
                    f"{pc:.4f}", f"{pw:.4f}", f"{dm:.4f}",
                    f"{pcp:.4f}" if pcp != "" else "", f"{pwp:.4f}" if pwp != "" else "",
                    f"{dp:.4f}" if dp != "" else "", bucket])

# ---- delta_distribution.csv (verifier ladder) --------------------------
with open(os.path.join(OUT, "delta_distribution.csv"), "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["verifier", "n_correct", "n_wrong", "p_correct", "p_wrong", "delta",
                "obs_required_80pct_power"])
    ladder = [
        ("mbpp_3_binary", graded, lambda r: float(r["passed"])),
        ("mbpp_3_graded", graded,
         lambda r: r["tests_passed"] / r["tests_total"] if r["tests_total"] else float(r["passed"])),
        ("mbpp_plus_binary", plus, lambda r: float(r["mbppplus_passed"])),
        ("mbpp_plus_graded", plus, lambda r: float(r["mbppplus_fraction"])),
    ]
    for name, src, fn in ladder:
        if not src:
            continue
        c = [fn(r) for r in src if own(r)]
        x = [fn(r) for r in src if not own(r)]
        d = diagnosticity(c, x)
        w.writerow([name, len(c), len(x), f"{d['p_correct']:.4f}", f"{d['p_wrong']:.4f}",
                    f"{d['delta']:.4f}", f"{d['n_required']:.1f}"])

# ---- convergence.csv (delta -> gain law) ------------------------------
LAW = [(0.05, 87.9), (0.10, 88.3), (0.20, 89.7), (0.245, 90.5),
       (0.40, 92.2), (0.60, 93.6), (0.80, 94.5), (1.00, 95.7)]
STATIC = 87.2
with open(os.path.join(OUT, "convergence.csv"), "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["delta", "final_hit_at_1", "static_baseline", "gain_pts", "obs_required"])
    for d, h in LAW:
        w.writerow([d, h, STATIC, f"{h-STATIC:.1f}", f"{observations_required(d):.1f}"])

for f in sorted(os.listdir(OUT)):
    p = os.path.join(OUT, f)
    print(f"{f:<26} {sum(1 for _ in open(p))-1:>4} rows")
