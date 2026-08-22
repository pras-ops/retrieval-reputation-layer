"""
Upgrade cached binary outcomes to graded ones, offline.

The cache already stores each completion, and MBPP ships several asserts per problem, so
"how many tests passed" is recoverable without any generation. That matters because the
observations a reputation layer needs to identify a document scale as 1/Delta^2, where
Delta is the pass-rate gap between retrieving the right evidence and the wrong evidence.
A binary verifier collapses partial credit and shrinks Delta; grading widens it for free.

Usage:  python3 sim/grade_cache.py [--limit N] [--workers K]
"""

import argparse
import json
import os
import re
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

DATA = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
MBPP = os.path.join(DATA, "mbpp.jsonl")
REAL = os.path.join(DATA, "gemini_cache.jsonl")
OUT = os.path.join(DATA, "gemini_cache_graded.jsonl")


def run_one(args):
    setup, completion, assertion = args
    program = (setup or "") + "\n" + completion + "\n" + assertion + "\n"
    try:
        proc = subprocess.run(
            [sys.executable, "-c", program], capture_output=True, text=True, timeout=10.0
        )
        return 1 if proc.returncode == 0 else 0
    except Exception:
        return 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    problems = {}
    for line in open(MBPP):
        p = json.loads(line)
        problems[p["task_id"]] = p

    rows = [json.loads(l) for l in open(REAL) if l.strip()]
    if args.limit:
        rows = rows[: args.limit]

    jobs, index = [], []
    for i, r in enumerate(rows):
        prob = problems.get(r["task_id"])
        if prob is None:
            continue
        for a in prob["test_list"]:
            jobs.append((prob.get("test_setup_code"), r["completion"], a))
            index.append(i)

    print(f"grading {len(rows)} cached completions over {len(jobs)} individual asserts...")
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        results = list(ex.map(run_one, jobs, chunksize=16))

    passed_count = [0] * len(rows)
    total_count = [0] * len(rows)
    for idx, ok in zip(index, results):
        total_count[idx] += 1
        passed_count[idx] += ok

    with open(OUT, "w") as fh:
        for i, r in enumerate(rows):
            out = dict(r)
            out["tests_passed"] = passed_count[i]
            out["tests_total"] = total_count[i]
            fh.write(json.dumps(out) + "\n")

    graded = [
        passed_count[i] / total_count[i] for i in range(len(rows)) if total_count[i]
    ]
    binary = [float(r["passed"]) for i, r in enumerate(rows) if total_count[i]]
    partial = sum(1 for g in graded if 0.0 < g < 1.0)
    print(f"wrote {OUT}")
    print(f"  mean binary outcome  {sum(binary)/len(binary):.4f}")
    print(f"  mean graded outcome  {sum(graded)/len(graded):.4f}")
    print(f"  rows with PARTIAL credit that binary scoring threw away: {partial} "
          f"({partial/len(graded):.1%})")


if __name__ == "__main__":
    main()
