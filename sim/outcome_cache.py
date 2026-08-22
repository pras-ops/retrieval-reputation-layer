"""
Provenance-tagged outcome cache.

A replay cache is only a reproducibility asset if every row states who produced it. When
mock and real rows share a file, a plumbing self-test can write synthetic labels that a
later replay reads as measurements — and mock labels are usually a perfect function of
whatever the experiment is trying to measure, so the contamination flatters the method
instead of adding noise.

Two rules enforced here:
  1. Mock and real outcomes live in different files. `--mock` cannot write to the real one.
  2. Every row carries a generator/verifier identity, and a strict reader refuses rows
     that lack it. Missing provenance is an error, never a silent default.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, Optional, Tuple
import hashlib
import json
import os

REAL_CACHE = "gemini_cache.jsonl"
MOCK_CACHE = "mock_cache.jsonl"

MOCK_GENERATORS = {"mock", "selftest", "stub"}


class CacheProvenanceError(RuntimeError):
    """Raised when a cache row cannot be trusted for a reported measurement."""


@dataclass
class OutcomeRecord:
    task_id: int
    retrieved_id: str
    retrieved_content: str
    completion: str
    passed: float
    # Provenance. Every field is required for a row to count as a real measurement.
    generator: str = ""
    generator_version: str = ""
    verifier: str = ""
    verifier_version: str = ""
    seed: Optional[int] = None
    timestamp: Optional[float] = None
    # Graded outcome, when the verifier can report more than pass/fail.
    tests_passed: Optional[int] = None
    tests_total: Optional[int] = None
    attribution: Optional[str] = None
    prompt_sha256: str = ""

    @property
    def is_mock(self) -> bool:
        return self.generator.lower() in MOCK_GENERATORS

    @property
    def fraction(self) -> float:
        """Graded outcome when available, else the binary one."""
        if self.tests_total:
            return float(self.tests_passed or 0) / float(self.tests_total)
        return float(self.passed)

    def key(self) -> Tuple[int, str]:
        return (self.task_id, self.retrieved_id)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


class OutcomeCache:
    """
    Keyed by (task_id, retrieved_id). Generation is deterministic at temperature 0, so a
    (problem, evidence) pair has one outcome and the cache is a complete lookup table
    rather than a sample.
    """

    def __init__(self, path: str, *, strict: bool = True, allow_mock: bool = False):
        self.path = path
        self.strict = strict
        self.allow_mock = allow_mock
        self.records: Dict[Tuple[int, str], OutcomeRecord] = {}
        self.rejected: Dict[str, int] = {}

    # ---- reading ---------------------------------------------------------

    def load(self) -> "OutcomeCache":
        self.records = {}
        self.rejected = {}
        if not os.path.exists(self.path):
            return self
        with open(self.path, "r") as fh:
            for line in fh:
                if not line.strip():
                    continue
                try:
                    raw = json.loads(line)
                except json.JSONDecodeError:
                    self._reject("unparseable")
                    continue
                rec = self._coerce(raw)
                if rec is None:
                    continue
                self.records[rec.key()] = rec
        if self.strict and self.rejected:
            raise CacheProvenanceError(
                f"{self.path}: refused {sum(self.rejected.values())} row(s) "
                f"({dict(self.rejected)}). Regenerate the cache or pass strict=False to "
                f"inspect it. A row without provenance cannot back a reported number."
            )
        return self

    def _reject(self, reason: str) -> None:
        self.rejected[reason] = self.rejected.get(reason, 0) + 1

    def _coerce(self, raw: dict) -> Optional[OutcomeRecord]:
        if "generator" not in raw or not raw.get("generator"):
            self._reject("missing_generator")
            return None
        known = {f for f in OutcomeRecord.__dataclass_fields__}
        rec = OutcomeRecord(**{k: v for k, v in raw.items() if k in known})
        if rec.is_mock and not self.allow_mock:
            self._reject("mock_generator")
            return None
        if not rec.verifier:
            self._reject("missing_verifier")
            return None
        return rec

    def get(self, task_id: int, retrieved_id: str) -> Optional[OutcomeRecord]:
        return self.records.get((task_id, retrieved_id))

    def outcome(self, task_id: int, retrieved_id: str, graded: bool = False) -> Optional[float]:
        rec = self.get(task_id, retrieved_id)
        if rec is None:
            return None
        return rec.fraction if graded else float(rec.passed)

    def __len__(self) -> int:
        return len(self.records)

    # ---- writing ---------------------------------------------------------

    def put(self, rec: OutcomeRecord) -> None:
        if rec.is_mock and not self.allow_mock:
            raise CacheProvenanceError(
                f"refusing to write a mock outcome into {self.path}. Mock runs must target "
                f"{MOCK_CACHE}; mixing them makes every replayed number unverifiable."
            )
        if not rec.generator or not rec.verifier:
            raise CacheProvenanceError("every cached outcome needs a generator and a verifier")
        self.records[rec.key()] = rec
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        with open(self.path, "a") as fh:
            fh.write(json.dumps(asdict(rec)) + "\n")

    # ---- reporting -------------------------------------------------------

    def summary(self) -> dict:
        gens: Dict[str, int] = {}
        for rec in self.records.values():
            gens[rec.generator] = gens.get(rec.generator, 0) + 1
        return {
            "path": self.path,
            "rows": len(self.records),
            "generators": gens,
            "rejected": dict(self.rejected),
        }


def cache_for(data_dir: str, *, mock: bool) -> OutcomeCache:
    """The real cache for measurements, the mock cache for plumbing checks. Never both."""
    if mock:
        return OutcomeCache(os.path.join(data_dir, MOCK_CACHE), strict=False, allow_mock=True)
    return OutcomeCache(os.path.join(data_dir, REAL_CACHE), strict=True, allow_mock=False)
