"""
RRL Core Logic Package
RAG Feedback and Exploration Loop implementation.
"""

from .store import Candidate, CandidateStore
from .retriever import Retriever
from .layer import ReputationLayer, RescoreResult
from .reliability import SourceReliability
from .drift import ChangeDetector
from .feedback import (
    OutcomeSignals,
    calculate_outcome,
    update_counters,
    update_counters_from_shares,
    update_counters_with_signals,
)

__all__ = [
    "Candidate",
    "CandidateStore",
    "Retriever",
    "ReputationLayer",
    "RescoreResult",
    "SourceReliability",
    "ChangeDetector",
    "OutcomeSignals",
    "calculate_outcome",
    "update_counters",
    "update_counters_from_shares",
    "update_counters_with_signals",
]
