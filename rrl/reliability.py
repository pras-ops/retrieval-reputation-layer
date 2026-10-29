"""
RRL Feedback-Source Reliability
Learns how often each unverified feedback source (user behavior, LLM judge, thumbs)
agrees with the verifier, and combines unverified signals by those learned accuracies.

The verifier (s_gt) is the root of trust, as the small clean root dataset is in FLTrust
(Cao et al., NDSS 2021). Each source is modeled as a one-coin annotator in the sense of
Dawid & Skene (1979): it reports the true outcome with accuracy a_s. Unverified signals are
then combined by a naive-Bayes log-odds vote with weight log(a_s / (1 - a_s)), so a source
that has never been checked against the verifier, or that agrees with it only at chance,
contributes nothing.

Accuracies are kept globally and per document, and the per-document estimate is shrunk toward
the global one (prior strength `doc_strength`). A source can be reliable in general and still
be fooled by one particular document — users who trust an authoritative-looking but wrong guide
— and only a per-document estimate can catch that.
"""

import math
from typing import Dict, Iterable, Optional, Tuple

from .feedback import OutcomeSignals

SOURCES: Tuple[str, ...] = ("s_behave", "s_judge", "s_expl")


class SourceReliability:
    def __init__(
        self,
        prior: Tuple[float, float] = (1.0, 1.0),
        max_weight: float = 4.0,
        doc_strength: float = 4.0,
    ):
        self.prior = prior
        self.max_weight = max_weight
        self.doc_strength = doc_strength
        # source -> [agreements, disagreements] with the verifier
        self.counts: Dict[str, list] = {s: [0.0, 0.0] for s in SOURCES}
        # doc id -> source -> [agreements, disagreements]
        self.doc_counts: Dict[str, Dict[str, list]] = {}

    def observe(self, signals: OutcomeSignals, doc_ids: Iterable[str] = ()) -> None:
        """Score every present source against the verifier, when the verifier is present."""
        if signals.s_gt is None:
            return
        truth = signals.s_gt > 0.5
        docs = list(doc_ids)
        for src in SOURCES:
            val = getattr(signals, src)
            if val is None:
                continue
            slot = 0 if (val > 0.5) == truth else 1
            self.counts[src][slot] += 1.0
            for d in docs:
                per_doc = self.doc_counts.setdefault(d, {s: [0.0, 0.0] for s in SOURCES})
                per_doc[src][slot] += 1.0

    def accuracy(self, src: str, doc_id: Optional[str] = None) -> float:
        """Posterior mean agreement with the verifier, per document when doc_id is given."""
        agree, disagree = self.counts[src]
        a0, b0 = self.prior
        acc = (agree + a0) / (agree + disagree + a0 + b0)
        if doc_id is None or doc_id not in self.doc_counts:
            return acc
        d_agree, d_disagree = self.doc_counts[doc_id][src]
        k = self.doc_strength
        return (d_agree + k * acc) / (d_agree + d_disagree + k)

    def weight(self, src: str, doc_id: Optional[str] = None) -> float:
        """Log-odds vote weight; zero for a source no better than chance."""
        a = self.accuracy(src, doc_id)
        if a <= 0.5:
            return 0.0
        return min(self.max_weight, math.log(a / (1.0 - a)))

    def combine(self, signals: OutcomeSignals, doc_id: Optional[str] = None) -> Optional[float]:
        """
        Outcome y in [0, 1]. The verifier overrides when present. Otherwise returns the
        posterior P(success) from a log-odds vote of the unverified sources, or None when
        no present source has earned any weight (the caller should then skip the update).
        """
        if signals.s_gt is not None:
            return max(0.0, min(1.0, signals.s_gt))
        logit = 0.0
        total = 0.0
        for src in SOURCES:
            val = getattr(signals, src)
            if val is None:
                continue
            w = self.weight(src, doc_id)
            logit += w if val > 0.5 else -w
            total += w
        if total == 0.0:
            return None
        return 1.0 / (1.0 + math.exp(-logit))

    def to_dict(self) -> dict:
        return {
            "prior": list(self.prior),
            "max_weight": self.max_weight,
            "doc_strength": self.doc_strength,
            "counts": self.counts,
            "doc_counts": self.doc_counts,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SourceReliability":
        obj = cls(
            prior=tuple(data["prior"]),  # type: ignore[arg-type]
            max_weight=data["max_weight"],
            doc_strength=data.get("doc_strength", 4.0),
        )
        obj.counts = {s: list(data["counts"].get(s, [0.0, 0.0])) for s in SOURCES}
        obj.doc_counts = {
            d: {s: list(c.get(s, [0.0, 0.0])) for s in SOURCES}
            for d, c in data.get("doc_counts", {}).items()
        }
        return obj
