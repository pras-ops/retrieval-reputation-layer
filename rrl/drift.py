"""
RRL Change Detection
Detects an abrupt change in a document's usefulness from its recent outcomes and resets its
short-term counters to the post-change evidence, instead of relying only on a fixed decay
rate. This is a two-window mean-shift test, in the spirit of sliding-window bandits
(Garivier & Moulines, ALT 2011) and adaptive windowing (ADWIN, Bifet & Gavaldà 2007).
"""

from .store import Candidate


class ChangeDetector:
    def __init__(self, window: int = 4, threshold: float = 0.6):
        """
        window:    outcomes in each of the two compared windows.
        threshold: absolute difference of window means that counts as a change.
        """
        self.window = window
        self.threshold = threshold

    def detect(self, candidate: Candidate) -> bool:
        outcomes = candidate.recent_outcomes
        w = self.window
        if len(outcomes) < 2 * w:
            return False
        old = sum(outcomes[-2 * w : -w]) / w
        new = sum(outcomes[-w:]) / w
        return abs(new - old) >= self.threshold

    def reset(self, candidate: Candidate) -> None:
        """Rebuild the short-term counters from the post-change window only."""
        recent = candidate.recent_outcomes[-self.window :]
        candidate.alpha = 1.0 + sum(recent)
        candidate.beta = 1.0 + sum(1.0 - y for y in recent)
        candidate.recent_outcomes = list(recent)

    def check(self, candidate: Candidate) -> bool:
        """Detect and, if a change is found, reset. Returns whether a reset happened."""
        if self.detect(candidate):
            self.reset(candidate)
            return True
        return False
