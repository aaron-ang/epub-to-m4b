"""Length-sorted windows across chapters for a single ``synthesize()`` call.

The orchestrator already knows every sentence still missing from the cache,
across every chapter, before it makes the first ``synthesize()`` call - there
is no unknown future arrival to buffer against. So the whole pending pool
can be sorted by text length once and sliced into ``max_batch``-sized
groups: a single call never mixes a five-word sentence with a
two-hundred-word one (which would waste the short slots in a fixed-size
batch), and ``max_batch`` is respected as a hard cap.

Longest first: the heaviest batch runs at the start, so a batch too big for
the GPU fails in the first minute rather than hours in, the progress log
only ever speeds up, and the likeliest runaways (long texts) fill the retry
queue early enough to go out as full batches.

Sorting loses chapter order on purpose. Nothing downstream needs it back:
every result is stored to the clip cache under its item's ``key`` the
moment it returns, and chapters are assembled later by looking their keys
up again.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PendingClip:
    """One not-yet-cached sentence and the key its clip is stored under."""

    key: str
    text: str


def make_batches(items: Sequence[PendingClip], max_batch: int) -> list[list[PendingClip]]:
    """Sort ``items`` by text length, longest first, then slice into groups of at most
    ``max_batch``. Order within/across batches carries no meaning beyond
    length-similarity."""
    if max_batch < 1:
        raise ValueError(f"max_batch must be >= 1, got {max_batch}")
    ordered = sorted(items, key=lambda item: len(item.text), reverse=True)
    return [list(ordered[i : i + max_batch]) for i in range(0, len(ordered), max_batch)]
