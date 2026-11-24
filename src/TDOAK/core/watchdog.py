"""Frame watchdog for the camera pump (worker thread, pure Python).

Detects the two failures that otherwise look healthy from the host: a pipeline that
starts but never delivers a frame (unsupported stream config, firmware crash right after
start) and a pipeline that stops delivering mid-run (device stall, brownout).
"""
from __future__ import annotations

import time
from typing import Callable, Iterable


class FrameWatchdog:
    """Call mark(stream) for every received message and check() on idle passes.

    firstFrameTimeout — seconds allowed between start and the first message on ANY stream.
    stallTimeout      — seconds allowed without a message on ANY stream once frames flowed.
    A timeout of 0 (or less) disables that check.
    """

    def __init__(self, firstFrameTimeout: float, stallTimeout: float,
                 streams: Iterable[str] = (),
                 clock: Callable[[], float] = time.monotonic) -> None:
        self.firstFrameTimeout = float(firstFrameTimeout)
        self.stallTimeout = float(stallTimeout)
        self._clock = clock
        self._start = clock()
        self._last: float | None = None
        self._lastPer: dict[str, float | None] = {name: None for name in streams}
        self._reported: set[str] = set()

    @property
    def anyFrame(self) -> bool:
        return self._last is not None

    def mark(self, stream: str) -> None:
        now = self._clock()
        self._last = now
        self._lastPer[stream] = now
        self._reported.discard(stream)

    def check(self) -> str | None:
        """Error text when the run should be failed, else None."""
        now = self._clock()
        if self._last is None:
            if self.firstFrameTimeout > 0 and now - self._start >= self.firstFrameTimeout:
                return (f'no frames within {self.firstFrameTimeout:g} s of start - the device '
                        'accepted the pipeline but sends nothing (unsupported stream '
                        'configuration, or the device stalled/crashed)')
            return None
        if self.stallTimeout > 0 and now - self._last >= self.stallTimeout:
            return (f'no frames for {self.stallTimeout:g} s - the device stopped sending '
                    '(stall, power or cable)')
        return None

    def ages(self) -> dict[str, float]:
        """Seconds since each stream's last message (since start for a stream never seen)."""
        now = self._clock()
        return {name: now - (last if last is not None else self._start)
                for name, last in self._lastPer.items()}

    def silentStreams(self) -> list[str]:
        """Streams that went quiet while others still flow — each is returned ONCE per
        silence (again only after it has resumed and stopped a second time). Empty while
        the stall check is disabled or no frame has arrived at all.

        A stream that has not delivered ANYTHING yet gets the (longer) first-frame
        allowance: sensors and host-side nodes warm up at different speeds (a ToF depth
        stream starts seconds after the colour stream), which is not a stall."""
        if self.stallTimeout <= 0 or self._last is None:
            return []
        now = self._clock()
        warmup = max(self.firstFrameTimeout, self.stallTimeout)
        fresh = []
        for name, last in self._lastPer.items():
            if name in self._reported:
                continue
            silent = (now - self._start >= warmup) if last is None else (
                now - last >= self.stallTimeout)
            if silent:
                fresh.append(name)
        self._reported.update(fresh)
        return fresh
