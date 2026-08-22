"""
RRL Clock abstraction.

Reputation decay is a function of *elapsed time*, so every component that reads or
writes counters needs to agree on what "now" means. Mixing bases (a wall-clock
timestamp stored on a candidate, a small integer step passed by a simulation) makes
`now - last_feedback` meaningless and silently disables decay. Passing an explicit
clock keeps production on wall time and experiments on simulated time without
either one touching the scoring logic.
"""

from typing import Optional
import time


class Clock:
    """Base clock. `now()` returns a float on a single, self-consistent time base."""

    unit_sec: float = 1.0

    def now(self) -> float:  # pragma: no cover - interface
        raise NotImplementedError


class WallClock(Clock):
    """Production clock: POSIX seconds. One decay unit defaults to a day."""

    def __init__(self, unit_sec: float = 86400.0):
        self.unit_sec = unit_sec

    def now(self) -> float:
        return time.time()


class SimClock(Clock):
    """
    Deterministic clock for simulations and tests. Starts at 0.0 and only moves when
    advanced, so a run is reproducible and `now - last_feedback` is a step count.
    """

    def __init__(self, start: float = 0.0, unit_sec: float = 1.0):
        self._t = float(start)
        self.unit_sec = unit_sec

    def now(self) -> float:
        return self._t

    def advance(self, dt: float = 1.0) -> float:
        self._t += float(dt)
        return self._t

    def set(self, t: float) -> float:
        self._t = float(t)
        return self._t


def resolve_now(clock: Optional[Clock], now: Optional[float]) -> Optional[float]:
    """
    Pick the timestamp to score against. An explicit `now` always wins so callers can
    replay a fixed trace; otherwise the clock is consulted. Returns None when neither
    is available, which callers treat as "do not decay".
    """
    if now is not None:
        return now
    if clock is not None:
        return clock.now()
    return None
