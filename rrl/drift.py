"""
RRL Change Detection
Detects an abrupt change in a document's usefulness from its recent outcomes and resets its
counters to the post-change evidence, instead of relying only on a fixed decay rate. This is a
two-window test, in the spirit of sliding-window bandits (Garivier & Moulines, ALT 2011) and
adaptive windowing (ADWIN, Bifet & Gavaldà 2007).

By default it is a two-proportion z-test on the last two windows at significance `alpha`, so
how big a shift must be to count scales with how noisy the document's outcomes are. A
document that works for some requests and not others has outcomes that flip a lot without
anything having changed; a fixed threshold resets it again and again (false resets), and the
test does not. Passing `threshold` instead uses a fixed difference of window means; with
binary outcomes that difference moves in steps of 1/window.

`resets` counts resets so that false resets under noisy-but-stationary feedback can be
measured. With a cluster id, the cluster's own counters are checked and reset.

By default (`history="verified"`) the test runs on the verifier's own 0/1 outcomes, which the
detector records through `observe()`: a proportion test needs binary observations, and the
learning targets in `recent_outcomes` are soft (an unverified success can count as 0.81). It
also means a tool whose unverified reports stay positive while its verified calls start failing
is still detected. `history="soft"` keeps the original behaviour, testing `recent_outcomes`.
"""

import math
from statistics import NormalDist
from typing import Dict, List, Optional, Tuple

from .store import Candidate


class ChangeDetector:
    def __init__(
        self,
        window: int = 8,
        threshold: Optional[float] = None,
        alpha: float = 0.01,
        reset_long_term: bool = False,
        history: str = "verified",
    ):
        """
        window:          outcomes in each of the two compared windows.
        threshold:       if given, a fixed difference of window means that counts as a change
                         (the original rule); otherwise a z-test at level alpha.
        alpha:           two-sided significance level of the z-test.
        reset_long_term: also reset the long-term A/B counters, so no pre-change evidence
                         remains in the score.
        history:         "verified" (test the verifier's 0/1 outcomes recorded by observe()) or
                         "soft" (test the soft learning targets in recent_outcomes).
        """
        self.window = window
        self.threshold = threshold
        self.alpha = alpha
        self.z_crit = NormalDist().inv_cdf(1.0 - alpha / 2.0)
        self.reset_long_term = reset_long_term
        if history not in ("verified", "soft"):
            raise ValueError(f"unknown history {history!r}")
        self.history = history
        self.verified: Dict[Tuple[str, Optional[str]], List[float]] = {}
        self.resets = 0

    def observe(self, candidate_id: str, success: bool, cluster_id: Optional[str] = None) -> None:
        """Record one verified outcome for a candidate (and for it within a cluster)."""
        keys = [(candidate_id, None)] + ([(candidate_id, cluster_id)] if cluster_id else [])
        for key in keys:
            h = self.verified.setdefault(key, [])
            h.append(1.0 if success else 0.0)
            del h[: -4 * self.window]

    def _outcomes(self, c, candidate_id: str, cluster_id: Optional[str]) -> List[float]:
        if self.history == "verified":
            return self.verified.get((candidate_id, cluster_id), [])
        if isinstance(c, dict):
            return c.get("recent_outcomes", [])
        return c.recent_outcomes

    def detect_outcomes(self, outcomes: List[float]) -> bool:
        w = self.window
        if len(outcomes) < 2 * w:
            return False
        old = sum(outcomes[-2 * w : -w]) / w
        new = sum(outcomes[-w:]) / w
        if self.threshold is not None:
            return abs(new - old) >= self.threshold
        pooled = (old + new) / 2.0
        if pooled <= 0.0 or pooled >= 1.0:
            return False
        se = math.sqrt(pooled * (1.0 - pooled) * 2.0 / w)
        return abs(new - old) / se >= self.z_crit

    def detect(self, candidate: Candidate) -> bool:
        return self.detect_outcomes(self._outcomes(candidate, candidate.id, None))

    def _reset_counters(self, c, recent: List[float]) -> None:
        """c is a Candidate or a cluster-counter dict."""
        values: Dict[str, object] = {
            "alpha": 1.0 + sum(recent),
            "beta": 1.0 + sum(1.0 - y for y in recent),
        }
        if self.history == "soft":
            values["recent_outcomes"] = list(recent)
        if self.reset_long_term:
            values["A"] = 1.0 + 0.25 * sum(recent)
            values["B"] = 1.0 + 0.25 * sum(1.0 - y for y in recent)
        for k, v in values.items():
            if isinstance(c, dict):
                c[k] = v
            else:
                setattr(c, k, v)

    def reset(self, candidate: Candidate) -> None:
        """Rebuild the counters from the post-change window only."""
        self._reset(candidate, candidate.id, None)

    def _reset(self, c, candidate_id: str, cluster_id: Optional[str]) -> None:
        recent = self._outcomes(c, candidate_id, cluster_id)[-self.window :]
        self._reset_counters(c, recent)
        if self.history == "verified":
            self.verified[(candidate_id, cluster_id)] = list(recent)

    def check(self, candidate: Candidate, cluster_id: Optional[str] = None) -> bool:
        """Detect and, if a change is found, reset. Returns whether a reset happened."""
        if cluster_id is not None:
            cc = getattr(candidate, "cluster_counters", {}).get(cluster_id)
            if cc is None:
                return False
            if not self.detect_outcomes(self._outcomes(cc, candidate.id, cluster_id)):
                return False
            self._reset(cc, candidate.id, cluster_id)
        else:
            if not self.detect(candidate):
                return False
            self.reset(candidate)
        self.resets += 1
        return True
