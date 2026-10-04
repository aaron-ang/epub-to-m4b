"""Percent-step gating for progress lines on long-running work."""

from __future__ import annotations

# A progress line is logged each time completion crosses another multiple of
# this many percent, plus once on reaching the total, so a run logs about
# 100 / step lines however many batches or seconds of audio it covers.
PROGRESS_STEP_PERCENT = 10


class PercentSteps:
    """``due(done)`` is true the first time ``done`` reaches each next
    ``PROGRESS_STEP_PERCENT`` of ``total``, and once when it reaches ``total``."""

    def __init__(self, total: float) -> None:
        self._total = total
        self._step = 0
        self._finished = False

    def due(self, done: float) -> bool:
        if self._finished:
            return False
        if done >= self._total:
            self._finished = True
            return True
        step = int(done * 100 // (self._total * PROGRESS_STEP_PERCENT))
        if step > self._step:
            self._step = step
            return True
        return False
