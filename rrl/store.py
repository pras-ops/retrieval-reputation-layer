"""
RRL Candidate and Store Implementations
Defines the Candidate dataclass with its Beta distribution counters and the CandidateStore.
"""

from dataclasses import dataclass, field
import datetime
from typing import Dict, List, Optional, Tuple
import time


def _decay(value: float, gamma: float, dt_units: float) -> float:
    """Beta-counter decay toward the prior of 1.0:  x <- 1 + (x-1) * gamma^dt."""
    if gamma >= 1.0 or dt_units <= 0:
        return value
    return 1.0 + (value - 1.0) * (gamma**dt_units)


# Sentinel for "no timestamp yet" in stores whose columns are NOT NULL.
# Both wall-clock and simulated time are non-negative, so -1.0 is unambiguous.
UNSET_TS = -1.0


def ts_to_db(value: Optional[float]) -> float:
    return UNSET_TS if value is None else float(value)


def ts_from_db(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    v = float(value)
    return None if v <= UNSET_TS else v


@dataclass
class Candidate:
    id: str
    content: str
    metadata: dict = field(default_factory=dict)

    # Short-term / Recent Usefulness Counters
    alpha: float = 1.0
    beta: float = 1.0

    # Permanent Usefulness Counters
    A: float = 1.0
    B: float = 1.0

    # Robustness & Denoising Fields
    fooled: float = 0.0
    verified: float = 0.0
    recent_outcomes: List[float] = field(default_factory=list)

    # Query-conditional counters (cluster_id -> dict of counters)
    cluster_counters: Dict[str, dict] = field(default_factory=dict)

    # Timestamp tracking.
    #
    # last_confirmed: when this candidate was last *confirmed useful* (positive outcome
    #   only). Semantic field, reported to callers; NOT the decay anchor.
    # last_feedback:  when this candidate last received an observation of any kind,
    #   success or failure. This is the decay anchor: evidence ages from the last time
    #   evidence arrived, so positive and negative evidence have equal lifetimes.
    # last_updated:   wall-clock creation/write stamp. Informational only — never used
    #   for decay, so it cannot leak wall time into a simulated run.
    #
    # All three start as None meaning "never happened". A None anchor means there is no
    # evidence to age, so decay is skipped. That is what keeps a candidate created under
    # one time base from being decayed against another.
    last_confirmed: Optional[float] = None
    last_feedback: Optional[float] = None
    last_updated: float = field(
        default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).timestamp()
    )

    @property
    def decay_anchor(self) -> Optional[float]:
        """
        Timestamp evidence ages from: the last observation, else the last confirmation,
        else the record's own write stamp.

        The final fallback is safe in both directions. A candidate ingested under wall
        time and scored under simulated time yields a negative dt, which is treated as
        "no elapsed time" rather than as decay; and a freshly ingested candidate sits at
        the Beta(1,1) prior, which decay leaves untouched by construction.
        """
        if self.last_feedback is not None:
            return self.last_feedback
        if self.last_confirmed is not None:
            return self.last_confirmed
        return self.last_updated

    def effective_counters(
        self,
        now: Optional[float] = None,
        gamma: float = 1.0,
        decay_unit_sec: float = 86400.0,
    ) -> Tuple[float, float, float, float]:
        """
        Decayed (alpha, beta, A, B) for scoring, computed without mutating the record.

        Decay must be a function of elapsed time, not of how many times a candidate was
        read. Returning values instead of writing them back makes repeated scoring
        idempotent: reading a document a hundred times between observations leaves its
        stored reputation exactly where the last observation left it.
        """
        anchor = self.decay_anchor
        if anchor is None or now is None or gamma >= 1.0 or decay_unit_sec <= 0:
            return self.alpha, self.beta, self.A, self.B
        dt = (now - anchor) / decay_unit_sec
        if dt <= 0:
            return self.alpha, self.beta, self.A, self.B
        return (
            _decay(self.alpha, gamma, dt),
            _decay(self.beta, gamma, dt),
            self.A,
            self.B,
        )

    def observations(self) -> float:
        """Total evidence mass on the short-term counters (Beta(1,1) prior = 0)."""
        return max(0.0, self.alpha + self.beta - 2.0)

    def get_cluster(self, cid: str) -> dict:
        """Returns the cluster dict, creating it with prior values if missing."""
        if cid not in self.cluster_counters:
            self.cluster_counters[cid] = {
                "alpha": 1.0,
                "beta": 1.0,
                "A": 1.0,
                "B": 1.0,
                "fooled": 0.0,
                "verified": 0.0,
                "recent_outcomes": [],
                "last_confirmed": self.last_confirmed,
                "last_feedback": self.last_feedback,
            }
        return self.cluster_counters[cid]

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "content": self.content,
            "metadata": self.metadata,
            "alpha": self.alpha,
            "beta": self.beta,
            "A": self.A,
            "B": self.B,
            "fooled": self.fooled,
            "verified": self.verified,
            "recent_outcomes": self.recent_outcomes,
            "cluster_counters": self.cluster_counters,
            "last_confirmed": self.last_confirmed,
            "last_feedback": self.last_feedback,
            "last_updated": self.last_updated,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Candidate":
        return cls(
            id=data["id"],
            content=data["content"],
            metadata=data.get("metadata", {}),
            alpha=data.get("alpha", 1.0),
            beta=data.get("beta", 1.0),
            A=data.get("A", 1.0),
            B=data.get("B", 1.0),
            fooled=data.get("fooled", 0.0),
            verified=data.get("verified", 0.0),
            recent_outcomes=data.get("recent_outcomes", []),
            cluster_counters=data.get("cluster_counters", {}),
            last_confirmed=data.get("last_confirmed"),
            last_feedback=data.get("last_feedback"),
            last_updated=data.get(
                "last_updated", datetime.datetime.now(datetime.timezone.utc).timestamp()
            ),
        )


class CandidateStore:
    def __init__(self):
        self.candidates: Dict[str, Candidate] = {}
        self.pending: Dict[str, Tuple[Dict[str, float], Optional[str], float]] = {}

    def add_candidate(self, candidate: Candidate) -> None:
        self.candidates[candidate.id] = candidate

    def get_candidate(self, candidate_id: str, now: Optional[float] = None) -> Optional[Candidate]:
        return self.candidates.get(candidate_id)

    def list_candidates(self, now: Optional[float] = None) -> List[Candidate]:
        return list(self.candidates.values())

    def update_candidate(self, candidate: Candidate) -> None:
        if candidate.id in self.candidates:
            self.candidates[candidate.id] = candidate
        else:
            raise KeyError(f"Candidate with ID {candidate.id} not found in store.")

    def save_pending(
        self,
        response_id: str,
        shares: Dict[str, float],
        cluster_id: Optional[str] = None,
        now: Optional[float] = None,
    ) -> None:
        if now is None:
            now = time.time()
        self.pending[response_id] = (shares, cluster_id, now)

    def pop_pending(self, response_id: str) -> Optional[Tuple[Dict[str, float], Optional[str]]]:
        if response_id in self.pending:
            shares, cluster_id, _ = self.pending.pop(response_id)
            return shares, cluster_id
        return None

    def gc_pending(self, max_age_sec: float, now: Optional[float] = None) -> int:
        if now is None:
            now = time.time()
        expired = [
            rid for rid, (_, _, created) in self.pending.items() if created < now - max_age_sec
        ]
        for rid in expired:
            self.pending.pop(rid)
        return len(expired)
