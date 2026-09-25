"""Deferred reseeds for clips the engine rejects, run as full batches.

An engine with ``retries`` above zero scores every clip with ``clip_miss``
(0.0 keeps it). A rejected clip is not retried on the spot: it waits here
with its best take so far until ``max_batch`` of them have piled up, and
then they are reseeded together through ``resynthesize`` - one full batch
instead of a string of one- or two-clip requests that leave the GPU mostly
idle. Whatever is still queued once the first pass is over goes out as a
final, possibly partial, batch.

Reseeds keep the engine's generation cap, so one runaway can't hold a whole
retry batch for long. A take the cap may have stopped mid-word scores
infinite and loses to any complete take. A text whose every capped take hit
the cap gets one last uncapped reseed, queued and batched the same way, so
the kept clip is never one the cap truncated.

Each flush is a new retry round, and the engine draws a fresh seed per
round, so a clip retried several times never repeats a take. Per clip the
take with the smallest miss wins, the earliest on a tie. Nothing is ever
cut: a clip still rejected after its last retry keeps its best take, with a
note.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from epub_to_m4b.book import AudioClip
from epub_to_m4b.synth.batching import PendingClip
from epub_to_m4b.tts.base import TTSEngine


@dataclass(slots=True)
class _Entry:
    item: PendingClip
    best: AudioClip
    miss: float
    attempts: int = 0


@dataclass(frozen=True, slots=True)
class Settled:
    """A clip ready for the cache, plus a note when it took retries."""

    item: PendingClip
    clip: AudioClip
    note: str | None = None


class RetryQueue:
    def __init__(self, engine: TTSEngine) -> None:
        self._engine = engine
        self._capped: list[_Entry] = []
        self._uncapped: list[_Entry] = []
        self._round = 0

    def __len__(self) -> int:
        return len(self._capped) + len(self._uncapped)

    def offer(self, item: PendingClip, clip: AudioClip) -> Settled | None:
        """Settle a first-pass clip now, or queue it for a reseed (``None``)."""
        if self._engine.retries < 1:
            return Settled(item, clip)
        miss = self._engine.clip_miss(item.text, clip, capped=True)
        if miss == 0.0:
            return Settled(item, clip)
        self._capped.append(_Entry(item, clip, miss))
        return None

    def flush(self, *, final: bool) -> list[Settled]:
        """Reseed queued clips in full ``max_batch`` groups; with ``final``,
        keep going with partial groups until both queues are empty."""
        size = self._engine.max_batch
        settled: list[Settled] = []
        while True:
            if len(self._capped) >= size or (final and self._capped):
                group, self._capped = self._capped[:size], self._capped[size:]
                settled.extend(self._retry(group, capped=True))
            elif len(self._uncapped) >= size or (final and self._uncapped):
                group, self._uncapped = self._uncapped[:size], self._uncapped[size:]
                settled.extend(self._retry(group, capped=False))
            else:
                return settled

    def _retry(self, group: list[_Entry], *, capped: bool) -> list[Settled]:
        self._round += 1
        texts = [entry.item.text for entry in group]
        clips = self._engine.resynthesize(texts, self._round, capped=capped)
        if len(clips) != len(group):
            raise ValueError(f"engine returned {len(clips)} retry clips for {len(group)} texts")
        settled: list[Settled] = []
        for entry, clip in zip(group, clips, strict=True):
            entry.attempts += 1
            miss = self._engine.clip_miss(entry.item.text, clip, capped=capped)
            if miss < entry.miss:
                entry.best, entry.miss = clip, miss
            if entry.miss > 0.0 and entry.attempts < self._engine.retries:
                self._capped.append(entry)
            elif math.isinf(entry.miss) and capped:
                # Every take so far hit the cap: one last try without it.
                self._uncapped.append(entry)
            else:
                settled.append(Settled(entry.item, entry.best, _note(entry)))
        return settled


def _note(entry: _Entry) -> str:
    preview = f"{entry.item.text[:60]!r}"
    if entry.miss == 0.0:
        return f"clip recovered after {entry.attempts} retries: {preview}"
    return (
        f"clip kept at {entry.best.seconds:.1f}s, {entry.miss:.1f}s outside its "
        f"expected length after {entry.attempts} retries: {preview}"
    )
