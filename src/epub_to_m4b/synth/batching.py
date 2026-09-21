"""Length-sorted windows across chapters for a single ``synthesize()`` call.

The orchestrator already knows every sentence still missing from the cache,
across every chapter, before it makes the first ``synthesize()`` call - there
is no unknown future arrival to buffer against, unlike a streaming admission
system. So the whole pending pool can simply be sorted by text length once
and sliced into ``max_batch``-sized groups: a single call never mixes a
five-word sentence with a two-hundred-word one (which would waste the short
slots in a fixed-size batch and make per-clip runaway-guard budgeting harder
to reason about), and ``max_batch`` is respected as a hard cap.

Sorting loses chapter order, so each pending sentence carries its own
``chapter_index``/``position`` (and cache ``key``) through the batch - the
orchestrator zips ``synthesize()`` results back to their originating
sentence by that identity, never by position in the batch.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PendingClip:
    """One not-yet-cached sentence, tagged with where it belongs.

    ``key`` doubles as the opaque id batching/orchestrator code zips results
    back with - it is already unique per (engine fingerprint, pipeline
    version, text), so no separate id is needed.
    """

    key: str
    chapter_index: int
    position: int
    text: str


def make_batches(items: Sequence[PendingClip], max_batch: int) -> list[list[PendingClip]]:
    """Sort ``items`` by text length, then slice into groups of at most
    ``max_batch``. Order within/across batches carries no meaning beyond
    length-similarity; callers reconstruct chapter order from each item's
    ``chapter_index``/``position``, not from batch position.
    """
    if max_batch < 1:
        raise ValueError(f"max_batch must be >= 1, got {max_batch}")
    ordered = sorted(items, key=lambda item: len(item.text))
    return [list(ordered[i : i + max_batch]) for i in range(0, len(ordered), max_batch)]
