"""
RRL Feedback-Source Reliability
Learns how well each unverified feedback source (user behavior, LLM judge, thumbs, a tool's
self-report) predicts the verifier, and combines unverified signals by that evidence.

The verifier (s_gt) is the root of trust, as the small clean root dataset is in FLTrust
(Cao et al., NDSS 2021). Each source is modeled as a two-coin annotator in the sense of
Dawid & Skene (1979): it has a sensitivity (says "worked" when it did) and a specificity
(says "failed" when it did). Agreement rate alone is not enough: when most outcomes succeed,
a source that always says "worked" agrees with the verifier most of the time while telling us
nothing. With two coins its "worked" vote carries log(sens / (1 - spec)) ≈ log(1 / 1) = 0.

Unverified signals are combined as a naive-Bayes posterior: the verifier's base rate plus each
source's log-likelihood ratio. Sources are treated as independent, which they need not be (a
user and a judge reading the same answer); max_weight caps any one source's influence.

Rates are kept globally and per document, and the per-document rates are shrunk toward the
global ones (prior strength `doc_strength`; None keeps only global rates). A source can be
reliable in general and still be fooled by one particular document, and only a per-document
estimate can catch that.
"""

import math
from typing import Dict, Iterable, Optional, Tuple

from .feedback import OutcomeSignals

SOURCES: Tuple[str, ...] = ("s_behave", "s_judge", "s_expl")

# Confusion counts per source: [true positives, false negatives, true negatives, false positives]
TP, FN, TN, FP = 0, 1, 2, 3


def _zero() -> Dict[str, list]:
    return {s: [0.0, 0.0, 0.0, 0.0] for s in SOURCES}


class SourceReliability:
    def __init__(
        self,
        prior: Tuple[float, float] = (1.0, 1.0),
        max_weight: float = 4.0,
        doc_strength: Optional[float] = 4.0,
    ):
        self.prior = prior
        self.max_weight = max_weight
        self.doc_strength = doc_strength
        self.counts: Dict[str, list] = _zero()
        self.doc_counts: Dict[str, Dict[str, list]] = {}
        # verified outcomes seen: [successes, failures]
        self.base: list = [0.0, 0.0]

    def observe(self, signals: OutcomeSignals, doc_ids: Iterable[str] = ()) -> None:
        """Score every present source against the verifier, when the verifier is present."""
        if signals.s_gt is None:
            return
        truth = signals.s_gt > 0.5
        self.base[0 if truth else 1] += 1.0
        docs = list(doc_ids) if self.doc_strength is not None else []
        for src in SOURCES:
            val = getattr(signals, src)
            if val is None:
                continue
            says = val > 0.5
            slot = (TP if says else FN) if truth else (FP if says else TN)
            self.counts[src][slot] += 1.0
            for d in docs:
                self.doc_counts.setdefault(d, _zero())[src][slot] += 1.0

    def rates(self, src: str, doc_id: Optional[str] = None) -> Tuple[float, float]:
        """Posterior-mean (sensitivity, specificity), per document when doc_id is given."""
        a0, b0 = self.prior
        tp, fn, tn, fp = self.counts[src]
        sens = (tp + a0) / (tp + fn + a0 + b0)
        spec = (tn + a0) / (tn + fp + a0 + b0)
        if doc_id is None or self.doc_strength is None or doc_id not in self.doc_counts:
            return sens, spec
        k = self.doc_strength
        dtp, dfn, dtn, dfp = self.doc_counts[doc_id][src]
        return (dtp + k * sens) / (dtp + dfn + k), (dtn + k * spec) / (dtn + dfp + k)

    def llr(self, src: str, says_success: bool, doc_id: Optional[str] = None) -> float:
        """
        Log-likelihood ratio of a vote. Zero for a source no better than chance
        (sens + spec <= 1): it is ignored rather than inverted, since an adversarial source
        can change its behavior.
        """
        sens, spec = self.rates(src, doc_id)
        if sens + spec <= 1.0:
            return 0.0
        r = math.log(sens / (1.0 - spec)) if says_success else math.log((1.0 - sens) / spec)
        return max(-self.max_weight, min(self.max_weight, r))

    def combine(self, signals: OutcomeSignals, doc_id: Optional[str] = None) -> Optional[float]:
        """
        Outcome y in [0, 1]. The verifier overrides when present. Otherwise returns the
        posterior P(success), or None when no present source carries any evidence (the
        caller should then skip the update rather than learn from the base rate alone).
        """
        if signals.s_gt is not None:
            return max(0.0, min(1.0, signals.s_gt))
        evidence = 0.0
        informative = False
        for src in SOURCES:
            val = getattr(signals, src)
            if val is None:
                continue
            r = self.llr(src, val > 0.5, doc_id)
            if r != 0.0:
                informative = True
            evidence += r
        if not informative:
            return None
        succ, fail = self.base
        prior_logit = math.log((succ + 1.0) / (fail + 1.0))
        return 1.0 / (1.0 + math.exp(-(prior_logit + evidence)))

    def to_dict(self) -> dict:
        return {
            "prior": list(self.prior),
            "max_weight": self.max_weight,
            "doc_strength": self.doc_strength,
            "counts": self.counts,
            "doc_counts": self.doc_counts,
            "base": self.base,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SourceReliability":
        obj = cls(
            prior=tuple(data["prior"]),  # type: ignore[arg-type]
            max_weight=data["max_weight"],
            doc_strength=data.get("doc_strength", 4.0),
        )
        obj.counts = {s: list(data["counts"].get(s, [0.0] * 4)) for s in SOURCES}
        obj.doc_counts = {
            d: {s: list(c.get(s, [0.0] * 4)) for s in SOURCES}
            for d, c in data.get("doc_counts", {}).items()
        }
        obj.base = list(data.get("base", [0.0, 0.0]))
        return obj
