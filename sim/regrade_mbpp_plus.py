"""
Phase 0: re-grade cached completions against MBPP+ instead of MBPP's three asserts.

Why this is the first move. Verifier diagnosticity is
    Delta = P(pass | correct evidence) - P(pass | wrong evidence)
and on MBPP we measured Delta = 0.245, capped by P(pass | wrong) = 0.517 -- the model
passes half the time on the wrong document. Some of those are *false passes*: code that
satisfies three asserts but is actually wrong. MBPP+ ships ~108 tests per problem instead
of 3, so it catches them. If false passes explain a meaningful share of that 0.517, the
verifier gets sharper for free -- no new corpus, no new generation, and the completions are
already cached.

It also fixes a measurement problem. At temperature 0 each (task, document) pair yields one
bit, so a per-task Delta cannot be estimated from a single generation. Graded over ~108
tests, the same single generation yields a fraction, which is a usable per-pair estimate.

No API calls. Usage:  python3 sim/regrade_mbpp_plus.py [--workers K]
"""

import argparse
import json
import os
import sys
DATA = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
IN = os.path.join(DATA, "gemini_cache_graded.jsonl")
OUT = os.path.join(DATA, "gemini_cache_mbppplus.jsonl")

_STATE = {}


def _groundtruth(problems):
    """
    Expected outputs for every MBPP+ input, by executing the canonical solution.

    evalplus caches this as a pickle, but a few MBPP+ problems return re.Match objects,
    which are unpicklable -- so its own cache write raises. Computing in-process and
    skipping the cache avoids that, at the cost of ~25s per run.
    """
    from evalplus.gen.util import trusted_exec

    out = {}
    for task_id, problem in problems.items():
        src = problem["prompt"] + problem["canonical_solution"]
        oracle = {}
        try:
            oracle["base"], oracle["base_time"] = trusted_exec(
                src, problem["base_input"], problem["entry_point"], record_time=True
            )
            oracle["plus"], oracle["plus_time"] = trusted_exec(
                src, problem["plus_input"], problem["entry_point"], record_time=True
            )
        except Exception:
            continue
        out[task_id] = oracle
    return out


def _init():
    from evalplus.data import get_mbpp_plus

    problems = get_mbpp_plus()
    _STATE["problems"] = problems
    _STATE["gt"] = _groundtruth(problems)


def _grade(row):
    """Return (base_passed, base_total, plus_passed, plus_total) for one cached completion."""
    from evalplus.eval import untrusted_check

    problems, gt = _STATE["problems"], _STATE["gt"]
    key = f"Mbpp/{row['task_id']}"
    prob = problems.get(key)
    if prob is None or key not in gt:
        return None
    exp = gt[key]
    # The cached completion is a full function definition; MBPP+ drives it by entry point.
    code = prob["prompt"] + "\n" + row["completion"] if False else row["completion"]

    out = {}
    for tag, inputs, expected, ref_t in (
        ("base", prob["base_input"], exp["base"], exp["base_time"]),
        ("plus", prob["plus_input"], exp["plus"], exp["plus_time"]),
    ):
        if not inputs:
            out[tag] = (0, 0)
            continue
        try:
            _status, details = untrusted_check(
                "mbpp",
                code,
                inputs,
                prob["entry_point"],
                expected=expected,
                atol=prob["atol"],
                ref_time=ref_t,
                fast_check=False,
                min_time_limit=1.0,
                gt_time_limit_factor=4.0,
            )
            passed = int(sum(1 for d in details if d))
            out[tag] = (passed, len(inputs))
        except Exception:
            out[tag] = (0, len(inputs))
    return (row, out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(IN) if l.strip()]
    if args.limit:
        rows = rows[: args.limit]
    print(f"re-grading {len(rows)} cached completions against MBPP+ ...", flush=True)

    # Serial by necessity: untrusted_check spawns its own guarded subprocess per check, so
    # wrapping it in a process pool tears down the pool.
    _init()
    results = []
    for i, row in enumerate(rows):
        r = _grade(row)
        if r is not None:
            results.append(r)
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(rows)}", flush=True)

    n_written = 0
    with open(OUT, "w") as fh:
        for row, out in results:
            bp, bt = out["base"]
            pp, pt = out["plus"]
            rec = dict(row)
            rec["mbppplus_base_passed"] = bp
            rec["mbppplus_base_total"] = bt
            rec["mbppplus_plus_passed"] = pp
            rec["mbppplus_plus_total"] = pt
            rec["mbppplus_passed"] = 1.0 if (bp == bt and pp == pt and bt + pt > 0) else 0.0
            rec["mbppplus_fraction"] = (bp + pp) / (bt + pt) if (bt + pt) else 0.0
            rec["verifier"] = "mbpp_plus"
            rec["verifier_version"] = "v0.2.0"
            fh.write(json.dumps(rec) + "\n")
            n_written += 1
    print(f"wrote {n_written} rows -> {OUT}")
    print(f"(dropped {len(rows)-n_written} rows whose task_id is absent from MBPP+'s 378)")


if __name__ == "__main__":
    main()
