"""
Split the legacy mixed cache into a quarantine and a provenance-tagged real cache.

The legacy file has no provenance field, so rows are classified by the mock generator's
two signatures, which are exact and mutually exclusive:

  * completion is byte-for-byte '    pass\\n'  -> the mock's "wrong evidence" branch
  * completion equals the retrieved document   -> the mock's "right evidence" branch

Everything else is treated as a real generation. That is a heuristic applied to a file
that should never have needed one; rows it keeps are tagged `generator_version=
"recovered-legacy"` so they can be told apart from a clean regeneration and are never
mistaken for audited provenance.

Usage:  python3 sim/quarantine_cache.py [--apply]
Without --apply it only reports.
"""

import argparse
import json
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from outcome_cache import MOCK_CACHE, REAL_CACHE  # noqa: E402

DATA = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
LEGACY = os.path.join(DATA, "mbpp_sweep_cache.jsonl")
QUARANTINE_DIR = os.path.join(DATA, "quarantine")

MOCK_STUB = "    pass\n"


def classify(row: dict) -> str:
    completion = row.get("completion", "")
    if completion == MOCK_STUB or completion.strip() == "pass":
        return "mock_stub"
    if completion.strip() and completion.strip() == row.get("retrieved_content", "").strip():
        return "mock_echo"
    return "real"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="write the split files")
    args = ap.parse_args()

    if not os.path.exists(LEGACY):
        print(f"no legacy cache at {LEGACY}; nothing to do")
        return

    rows = [json.loads(line) for line in open(LEGACY) if line.strip()]
    buckets = {"real": [], "mock_stub": [], "mock_echo": []}
    for r in rows:
        buckets[classify(r)].append(r)

    total = len(rows)
    mock = len(buckets["mock_stub"]) + len(buckets["mock_echo"])
    print(f"legacy cache: {LEGACY}")
    print(f"  rows                      {total}")
    print(f"  mock 'pass' stub branch   {len(buckets['mock_stub'])}")
    print(f"  mock echo branch          {len(buckets['mock_echo'])}")
    print(f"  presumed real             {len(buckets['real'])}")
    print(f"  contaminated fraction     {mock / total:.1%}")

    for name, key in (("stub", "mock_stub"), ("echo", "mock_echo")):
        labels = {r.get("passed") for r in buckets[key]}
        if buckets[key]:
            print(f"  {name} branch labels        {sorted(labels)} (mock labels are deterministic)")

    if not args.apply:
        print("\nreport only. re-run with --apply to write the split.")
        return

    os.makedirs(QUARANTINE_DIR, exist_ok=True)
    q_path = os.path.join(QUARANTINE_DIR, "mock_contaminated.jsonl")
    with open(q_path, "w") as fh:
        for key in ("mock_stub", "mock_echo"):
            for r in buckets[key]:
                fh.write(json.dumps({**r, "generator": "mock", "quarantine_reason": key}) + "\n")

    real_path = os.path.join(DATA, REAL_CACHE)
    with open(real_path, "w") as fh:
        for r in buckets["real"]:
            fh.write(
                json.dumps(
                    {
                        **r,
                        "generator": "gemini-2.5-flash",
                        # Recovered by signature, not by recorded provenance. Do not treat
                        # these as audited: regenerate before publishing a number.
                        "generator_version": "recovered-legacy",
                        "verifier": "mbpp_unit_tests",
                        "verifier_version": "recovered-legacy",
                    }
                )
                + "\n"
            )

    mock_path = os.path.join(DATA, MOCK_CACHE)
    if not os.path.exists(mock_path):
        open(mock_path, "w").close()

    legacy_moved = os.path.join(QUARANTINE_DIR, "mbpp_sweep_cache.legacy.jsonl")
    os.replace(LEGACY, legacy_moved)

    print(f"\nwrote {len(buckets['real'])} recovered rows -> {real_path}")
    print(f"wrote {mock} contaminated rows      -> {q_path}")
    print(f"moved the legacy mixed file        -> {legacy_moved}")
    print("\nRecovered rows are tagged 'recovered-legacy'. Regenerate with real")
    print("generation before publishing any pass-rate number from them.")


if __name__ == "__main__":
    main()
