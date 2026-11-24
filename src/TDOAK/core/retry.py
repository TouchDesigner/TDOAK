"""Reconnect policy for the camera pump (pure Python, main thread)."""
from __future__ import annotations

from typing import Sequence

MODES = ('off', 'limited', 'forever')


class RetryPolicy:
    """Decides whether and when to restart after a failed run.

    mode 'off'      — never retry.
    mode 'limited'  — up to maxTries start attempts in a row, then give up until Start.
    mode 'forever'  — keep retrying, backing off up to the last backoff value.
    A run that streamed for at least healthyAfter seconds counts as healthy: the failure
    counter restarts, so rare drops over a long session never add up to a give-up.
    """

    def __init__(self, mode: str = 'limited', maxTries: int = 3, healthyAfter: float = 30.0,
                 backoff: Sequence[float] = (5.0, 10.0, 20.0, 40.0, 60.0)) -> None:
        self.mode = mode if mode in MODES else 'limited'
        self.maxTries = int(maxTries)
        self.healthyAfter = float(healthyAfter)
        self.backoff = tuple(float(b) for b in backoff)
        self.failures = 0

    def reset(self) -> None:
        self.failures = 0

    def onFailure(self, ranFor: float = 0.0) -> float | None:
        """Record a failed run that streamed for ranFor seconds. Returns the delay in
        seconds before the next attempt, or None to give up."""
        if ranFor >= self.healthyAfter:
            self.failures = 0
        self.failures += 1
        if self.mode == 'off':
            return None
        if self.mode == 'limited' and self.failures >= self.maxTries:
            return None
        return self.backoff[min(self.failures - 1, len(self.backoff) - 1)]

    def describe(self) -> str:
        """Short text for the Status readout, e.g. 'attempt 2 of 3'."""
        if self.mode == 'limited':
            return f'attempt {self.failures + 1} of {self.maxTries}'
        return f'attempt {self.failures + 1}'
