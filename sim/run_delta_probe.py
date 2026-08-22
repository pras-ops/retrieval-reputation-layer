"""
Phase 2: measure per-task verifier diagnosticity with a real generation probe.

Delta_q = P(solve | correct evidence) - P(solve | wrong evidence), per task.

Two design constraints that are easy to get wrong:

  * Temperature must be > 0. At temperature 0 a (task, document) pair yields one
    deterministic outcome, so a per-task *rate* is just its own single sample -- the
    distribution collapses onto 0 and 1 and no bucketing is possible. We draw k samples
    per pair at temperature 0.8.
  * The samples must be split. Delta_q is estimated on the first half and RRL is evaluated
    on the second half. Estimating and evaluating on the same samples is selection on the
    dependent variable: high-Delta buckets would carry their own selection noise into the
    result and inflate any gain.

Every candidate document is generated for, so the downstream evaluation has full outcome
coverage and never has to impute a missing result.

Outcomes are written to data/outcomes/probe/, which is separate from the frozen benchmark
definition in data/benchmark/ and from any replay cache. Resumable: existing keys are skipped.

Usage:
    python3 sim/run_delta_probe.py --tasks 5 --samples 2 --pilot     # cost/latency probe
    python3 sim/run_delta_probe.py --tasks 100 --samples 8
"""

import argparse
import json
import os
import random
import re
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from rrl.judge import _get_client

try:
    from google.genai import types
except ImportError:
    types = None

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BENCH = os.path.join(ROOT, "data", "benchmark")
OUT_DIR = os.path.join(ROOT, "data", "outcomes", "probe")
OUT = os.path.join(OUT_DIR, "delta_probe.jsonl")

MODEL = "gemini-2.5-flash"
TEMPERATURE = 0.8
GENERATOR_VERSION = "vertex-2026-08"

# Thinking budget is a treatment variable, not just a cost knob. Extended reasoning lets the
# model solve these tasks without consulting the retrieved example, which raises
# P(solve | wrong evidence) and therefore *lowers* verifier diagnosticity. It is also where
# almost all the output tokens go. Probing both settings measures the trade directly.
THINKING_BUDGET = None  # set from CLI; 0 disables thinking, None leaves the model default

_lock = threading.Lock()
_usage = {"in": 0, "out": 0, "calls": 0, "errors": 0, "retries": 0}
_err_counts = {}

# Vertex enforces a per-project per-minute request quota. Saturating it returns 429
# RESOURCE_EXHAUSTED, which at high concurrency silently becomes a near-total failure rate --
# a first attempt at 16 threads lost 82% of calls. Throttle, then retry with backoff, and
# record the error text so a quota problem is never mistaken for a model or data problem.
_rate_lock = threading.Lock()
_last_call = [0.0]
MIN_INTERVAL = 0.0  # seconds between request starts; set from --rps


def _throttle():
    if MIN_INTERVAL <= 0:
        return
    with _rate_lock:
        now = time.time()
        wait = _last_call[0] + MIN_INTERVAL - now
        if wait > 0:
            time.sleep(wait)
            now = time.time()
        _last_call[0] = now


def _is_retryable(exc) -> bool:
    text = f"{type(exc).__name__} {exc}".lower()
    return any(
        k in text
        for k in ("429", "resource_exhausted", "quota", "rate limit", "503", "unavailable",
                  "504", "deadline", "500", "internal")
    )


def load_bench():
    tasks = [json.loads(l) for l in open(os.path.join(BENCH, "rrl_tasks.jsonl")) if l.strip()]
    docs = {
        d["doc_id"]: d
        for d in (
            json.loads(l) for l in open(os.path.join(BENCH, "rrl_candidates.jsonl")) if l.strip()
        )
    }
    return tasks, docs


def entry_point(task):
    m = re.search(r"assert\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", task["tests"][0])
    return m.group(1) if m else ""


def run_tests(task, completion, timeout=10.0):
    """Fraction of the task's asserts that pass, plus the strict all-or-nothing outcome."""
    setup = task.get("test_setup_code") or ""
    passed = 0
    for a in task["tests"]:
        program = setup + "\n" + completion + "\n" + a + "\n"
        try:
            p = subprocess.run(
                [sys.executable, "-c", program], capture_output=True, text=True, timeout=timeout
            )
            passed += 1 if p.returncode == 0 else 0
        except Exception:
            pass
    total = len(task["tests"])
    return passed, total


def generate(client, task, doc_text, seed, max_attempts=6):
    prompt = (
        "You are a Python coding assistant. Using the reference example only if it is "
        "relevant, write a correct solution to the problem. Return ONLY the function "
        "definition, no markdown, no commentary.\n\n"
        f"Reference example:\n{doc_text}\n\n"
        f"Problem: {task['query']}\n"
        f"The function must be named `{entry_point(task)}`.\n"
    )
    kwargs = dict(temperature=TEMPERATURE, seed=seed, max_output_tokens=4096)
    if THINKING_BUDGET is not None:
        kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=THINKING_BUDGET)
    cfg = types.GenerateContentConfig(**kwargs)

    delay = 2.0
    last = None
    for attempt in range(max_attempts):
        _throttle()
        try:
            r = client.models.generate_content(model=MODEL, contents=prompt, config=cfg)
            break
        except Exception as exc:  # noqa: BLE001 - classified below
            last = exc
            if not _is_retryable(exc) or attempt == max_attempts - 1:
                raise
            with _lock:
                _usage["retries"] += 1
            time.sleep(delay + random.random())
            delay = min(delay * 2, 60.0)
    else:  # pragma: no cover
        raise last
    u = getattr(r, "usage_metadata", None)
    with _lock:
        _usage["calls"] += 1
        if u:
            _usage["in"] += u.prompt_token_count or 0
            _usage["out"] += (u.candidates_token_count or 0) + (
                getattr(u, "thoughts_token_count", 0) or 0
            )
    t = (r.text or "").strip()
    for fence in ("```python", "```"):
        if t.startswith(fence):
            t = t[len(fence) :]
    if t.endswith("```"):
        t = t[:-3]
    return t.strip()


def existing_keys():
    keys = set()
    if os.path.exists(OUT):
        for line in open(OUT):
            if line.strip():
                try:
                    d = json.loads(line)
                    keys.add((d["task_id"], d["doc_id"], d["sample_idx"]))
                except Exception:
                    pass
    return keys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", type=int, default=100)
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--concurrency", type=int, default=6)
    ap.add_argument("--rps", type=float, default=4.0,
                    help="max request starts per second across all threads (0 = unthrottled)")
    ap.add_argument("--pilot", action="store_true", help="report cost/latency and stop")
    ap.add_argument("--seed", type=int, default=20260822)
    ap.add_argument("--thinking-budget", type=int, default=None,
                    help="0 disables model thinking; omit for the model default")
    ap.add_argument("--out", default="", help="override output filename inside data/outcomes/probe")
    args = ap.parse_args()

    global THINKING_BUDGET, OUT, MIN_INTERVAL
    THINKING_BUDGET = args.thinking_budget
    MIN_INTERVAL = (1.0 / args.rps) if args.rps and args.rps > 0 else 0.0
    if args.out:
        OUT = os.path.join(OUT_DIR, args.out)

    tasks, docs = load_bench()
    rng = random.Random(args.seed)
    chosen = sorted(tasks, key=lambda t: t["task_id"])
    rng.shuffle(chosen)
    chosen = chosen[: args.tasks]

    client = _get_client()
    if client is None or types is None:
        sys.exit(
            "no generation client. Vertex needs GCP_PROJECT_ID + ADC "
            "(gcloud auth application-default login), or set GEMINI_API_KEY."
        )

    done = existing_keys()
    jobs = []
    for t in chosen:
        for doc_id in t["candidate_doc_ids"]:
            for s in range(args.samples):
                if (t["task_id"], doc_id, s) not in done:
                    jobs.append((t, doc_id, s))
    print(
        f"tasks={len(chosen)} candidates/task={len(chosen[0]['candidate_doc_ids'])} "
        f"samples={args.samples} temperature={TEMPERATURE}"
    )
    print(f"generations needed: {len(jobs)}  (already have {len(done)})")
    os.makedirs(OUT_DIR, exist_ok=True)

    t0 = time.time()
    fh = open(OUT, "a")

    def work(job):
        t, doc_id, s = job
        try:
            comp = generate(client, t, docs[doc_id]["text"], seed=s)
        except Exception as e:
            label = type(e).__name__
            txt = str(e)
            for code in ("429", "503", "504", "500"):
                if code in txt:
                    label = f"{label}:{code}"
                    break
            with _lock:
                _usage["errors"] += 1
                _err_counts[label] = _err_counts.get(label, 0) + 1
                if len(_err_counts) <= 8 and _err_counts[label] == 1:
                    print(f"    [first {label}] {txt[:220]}", flush=True)
            return {"task_id": t["task_id"], "doc_id": doc_id, "sample_idx": s, "error": txt[:300]}
        p, tot = run_tests(t, comp)
        return {
            "task_id": t["task_id"],
            "mbpp_task_id": t["mbpp_task_id"],
            "doc_id": doc_id,
            "is_correct_doc": doc_id == t["correct_doc_id"],
            "sample_idx": s,
            "completion": comp,
            "tests_passed": p,
            "tests_total": tot,
            "passed": 1.0 if (tot and p == tot) else 0.0,
            "fraction": (p / tot) if tot else 0.0,
            "generator": MODEL,
            "generator_version": GENERATOR_VERSION,
            "temperature": TEMPERATURE,
            "thinking_budget": THINKING_BUDGET,
            "verifier": "mbpp_unit_tests",
            "verifier_version": "coderag-bench-tests",
            "benchmark": "rrl_retrieval_benchmark@1.0.0",
        }

    written = 0
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = [ex.submit(work, j) for j in jobs]
        for i, f in enumerate(as_completed(futs)):
            rec = f.result()
            if "error" not in rec:
                with _lock:
                    fh.write(json.dumps(rec) + "\n")
                    written += 1
            if (i + 1) % 50 == 0:
                el = time.time() - t0
                rate = (i + 1) / el
                print(
                    f"  {i+1}/{len(jobs)}  {rate:.1f}/s  eta {(len(jobs)-i-1)/max(rate,1e-9)/60:.1f}m"
                    f"  errors={_usage['errors']}",
                    flush=True,
                )
            if args.pilot and (i + 1) >= min(len(jobs), args.tasks * args.samples * 7):
                break
    fh.close()
    el = time.time() - t0

    # Vertex Gemini 2.5 Flash list price at time of writing, USD per million tokens.
    p_in, p_out = 0.30, 2.50
    cost = _usage["in"] / 1e6 * p_in + _usage["out"] / 1e6 * p_out
    print(f"\ncalls={_usage['calls']} errors={_usage['errors']} retries={_usage['retries']} "
          f"written={written} in {el/60:.1f}m")
    if _err_counts:
        print("error breakdown:", dict(sorted(_err_counts.items(), key=lambda kv: -kv[1])))
    print(f"tokens in={_usage['in']:,} out={_usage['out']:,}")
    print(f"cost this run ~ ${cost:.2f}  ({_usage['calls'] and cost/_usage['calls']*1000:.3f} $/1k calls)")
    if args.pilot and _usage["calls"]:
        per = cost / _usage["calls"]
        rate = _usage["calls"] / el
        for n_tasks, n_s in ((100, 8), (297, 8)):
            total = n_tasks * 7 * n_s
            print(
                f"  projected {n_tasks} tasks x 7 docs x {n_s} samples = {total:,} calls "
                f"~ ${per*total:.2f}, ~{total/rate/60:.0f} min"
            )


if __name__ == "__main__":
    main()
