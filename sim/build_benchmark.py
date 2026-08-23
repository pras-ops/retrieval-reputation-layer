"""
Phase 1: freeze an RRL retrieval benchmark on externally annotated ground truth.

Ground truth comes from CodeRAG-Bench (arXiv:2406.14497), which ships a *manually
annotated* canonical document per programming problem plus a retrieval corpus of canonical
solutions. Using someone else's annotation is the point: a corpus we label ourselves invites
the obvious objection that the benchmark was built to suit the algorithm.

What is ours, and declared as such, is the distractor construction. Each task gets a fixed
5-candidate set:

  1 canonical (externally annotated)
  2 tier-1    same topic family, different task        -- lexically close, genuinely wrong
  1 tier-2    different family, still a Python solution -- semantically related
  1 tier-3    a HumanEval solution                      -- irrelevant

Fixing the candidate set removes first-stage variance so every arm ranks *identical*
candidates, which is the comparison the old harness got wrong. A second condition
(`--open`) instead retrieves candidates from the whole corpus, for the case where you want
first-stage effects included.

Three files are written, and the separation is deliberate: a benchmark definition that
never changes must not live in the same file as mutable observed outcomes. That is the
structural fix for the contamination that invalidated the previous results.

Usage:  python3 sim/build_benchmark.py
"""

import hashlib
import json
import os
import random
import re
from collections import defaultdict

OUT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "benchmark"))
SEED = 20260822
N_TIER1, N_TIER2, N_TIER3 = 4, 1, 1
DOCSTRING_QUOTES = ('"""', "'''")

STOP = {
    "write", "a", "an", "the", "to", "of", "function", "python", "program", "given",
    "find", "for", "that", "in", "is", "check", "whether", "from", "and", "using",
    "get", "return",
}


def topic(text: str) -> str:
    for w in re.findall(r"[a-z]+", (text or "").lower()):
        if w not in STOP:
            return w
    return "misc"


def strip_query_leakage(text: str) -> str:
    """
    Remove the leading natural-language description from a document.

    CodeRAG-Bench's MBPP solutions are prefixed with the problem statement as a comment,
    and HumanEval solutions carry it as a docstring. Left in place, the "correct" document
    literally contains the query, so retrieval degenerates into string matching: BM25 alone
    scored 99.7% Hit@1 on the first build and no reranking signal was measurable at all.
    Stripping it uniformly from every document -- correct and distractor alike, so no arm is
    advantaged -- forces retrieval to work from code semantics, which is the task we mean
    to measure.
    """
    lines = text.split("\n")
    i = 0
    while i < len(lines) and (lines[i].strip().startswith("#") or not lines[i].strip()):
        i += 1
    body = "\n".join(lines[i:])

    # Drop function docstrings, which restate the problem for HumanEval-sourced docs.
    kept, in_doc, quote = [], False, None
    for line in body.split("\n"):
        t = line.strip()
        if not in_doc:
            opener = next((q for q in DOCSTRING_QUOTES if t.startswith(q)), None)
            if opener is not None:
                if len(t) > 3 and t.endswith(opener):
                    continue  # single-line docstring
                in_doc, quote = True, opener
                continue
            kept.append(line)
        else:
            if t.endswith(quote):
                in_doc = False
            continue
    result = "\n".join(kept).strip()
    return result if result else body.strip()


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    from datasets import load_dataset

    mbpp = load_dataset("code-rag-bench/mbpp")["train"]
    sols = load_dataset("code-rag-bench/programming-solutions")["train"]
    os.makedirs(OUT, exist_ok=True)

    # ---- candidate corpus -------------------------------------------------
    docs = []
    by_task = {}
    for i in range(len(sols)):
        row = sols[i]
        meta = row["meta"] or {}
        src = meta.get("task_name", "unknown")
        tid = str(meta.get("task_id", ""))
        doc_id = f"{src}_{tid}_{i}"
        rec = {
            "doc_id": doc_id,
            "text": strip_query_leakage(row["text"]),
            "text_original": row["text"],
            "title": row.get("title") or "",
            "source": src,
            "source_task_id": tid,
        }
        docs.append(rec)
        if src == "mbpp" and tid:
            by_task.setdefault(tid, doc_id)

    cand_path = os.path.join(OUT, "rrl_candidates.jsonl")
    with open(cand_path, "w") as fh:
        for d in docs:
            fh.write(json.dumps(d) + "\n")

    doc_by_id = {d["doc_id"]: d for d in docs}
    mbpp_docs = [d for d in docs if d["source"] == "mbpp"]
    he_docs = [d for d in docs if d["source"] == "humaneval"]

    # topic family of each mbpp doc, taken from its own task text where available
    task_text = {str(mbpp[i]["task_id"]): mbpp[i]["text"] for i in range(len(mbpp))}
    fam_of_doc = {d["doc_id"]: topic(task_text.get(d["source_task_id"], d["title"])) for d in mbpp_docs}
    by_family = defaultdict(list)
    for d in mbpp_docs:
        by_family[fam_of_doc[d["doc_id"]]].append(d["doc_id"])

    # ---- tasks ------------------------------------------------------------
    rng = random.Random(SEED)
    tasks, skipped = [], 0
    for i in range(len(mbpp)):
        row = mbpp[i]
        tid = str(row["task_id"])
        correct = by_task.get(tid)
        if correct is None:
            skipped += 1
            continue
        fam = topic(row["text"])

        pool1 = [x for x in by_family.get(fam, []) if x != correct]
        pool2 = [d["doc_id"] for d in mbpp_docs if fam_of_doc[d["doc_id"]] != fam]
        pool3 = [d["doc_id"] for d in he_docs]
        if len(pool1) < N_TIER1 or len(pool2) < N_TIER2 or len(pool3) < N_TIER3:
            skipped += 1
            continue

        t1 = rng.sample(pool1, N_TIER1)
        t2 = rng.sample(pool2, N_TIER2)
        t3 = rng.sample(pool3, N_TIER3)
        cands = [correct] + t1 + t2 + t3
        rng.shuffle(cands)
        tasks.append(
            {
                "task_id": f"mbpp_{tid}",
                "mbpp_task_id": int(tid),
                "query": row["text"],
                "topic_family": fam,
                "correct_doc_id": correct,
                "candidate_doc_ids": cands,
                "distractor_tiers": {
                    "tier1_same_family": t1,
                    "tier2_other_family": t2,
                    "tier3_irrelevant": t3,
                },
                "tests": list(row["test_list"]),
                "test_setup_code": row.get("test_setup_code") or "",
                "reference_code": row["code"],
                # evidence_required is deliberately absent: it must be MEASURED by a
                # correct-vs-wrong probe, never asserted at construction time.
                "source_dataset": "CodeRAG-Bench/mbpp",
                "split": "test",
            }
        )

    task_path = os.path.join(OUT, "rrl_tasks.jsonl")
    with open(task_path, "w") as fh:
        for t in tasks:
            fh.write(json.dumps(t) + "\n")

    manifest = {
        "name": "rrl_retrieval_benchmark",
        "version": "1.0.0",
        "build_seed": SEED,
        "ground_truth_source": {
            "dataset": "code-rag-bench/mbpp",
            "paper": "arXiv:2406.14497",
            "note": "canonical document per problem is manually annotated upstream",
        },
        "corpus_source": {"dataset": "code-rag-bench/programming-solutions"},
        "verifier_sources": {
            "mbpp": "3 asserts per problem (from CodeRAG-Bench task rows)",
            "mbpp_plus": "evalplus MbppPlus v0.2.0, ~108 tests per problem",
        },
        "construction": {
            "query_leakage_stripped": True,
            "leakage_note": (
                "upstream canonical docs prefix the problem statement as a comment or "
                "docstring; removed uniformly from all documents, else BM25 alone scores "
                "99.7% Hit@1 and no reranking signal is measurable"
            ),
            "candidates_per_task": 1 + N_TIER1 + N_TIER2 + N_TIER3,
            "tier1_same_topic_family": N_TIER1,
            "tier2_other_family": N_TIER2,
            "tier3_irrelevant_humaneval": N_TIER3,
            "declared_by": "this repository (distractors only; correct doc is external)",
        },
        "counts": {"tasks": len(tasks), "candidate_docs": len(docs), "skipped_tasks": skipped},
        "files": {
            "rrl_tasks.jsonl": sha256_file(task_path),
            "rrl_candidates.jsonl": sha256_file(cand_path),
        },
        "invariant": "definition files are immutable; observed outcomes live in data/outcomes/",
    }
    man_path = os.path.join(OUT, "manifest.json")
    with open(man_path, "w") as fh:
        json.dump(manifest, fh, indent=2)

    print(f"tasks:            {len(tasks)}  (skipped {skipped})")
    print(f"candidate docs:   {len(docs)}  ({len(mbpp_docs)} mbpp, {len(he_docs)} humaneval)")
    print(f"candidates/task:  {1+N_TIER1+N_TIER2+N_TIER3}")
    print(f"wrote {task_path}")
    print(f"wrote {cand_path}")
    print(f"wrote {man_path}")
    print(f"tasks sha256:      {manifest['files']['rrl_tasks.jsonl'][:16]}...")
    print(f"candidates sha256: {manifest['files']['rrl_candidates.jsonl'][:16]}...")


if __name__ == "__main__":
    main()
