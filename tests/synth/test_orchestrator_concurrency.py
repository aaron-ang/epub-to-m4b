"""synthesize_book's fan-out for engines that allow more than one request in
flight: batches really overlap, results land under the right keys whatever
order they finish in, and a failing batch ends the run."""

from __future__ import annotations

import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar

import numpy as np
import pytest

from epub_to_m4b.book import AudioClip, Book, Chapter, Paragraph, ParagraphKind
from epub_to_m4b.synth import cache
from epub_to_m4b.synth.orchestrator import synthesize_book
from epub_to_m4b.text import TEXT_PIPELINE_VERSION
from epub_to_m4b.tts.base import TTSEngine
from tests.helpers import make_book

_SHA = "book-sha-concurrency"


def _sentence(i: int) -> str:
    # Distinct lengths per sentence so each clip's sample count identifies its text.
    return f"Sentence number {i} keeps going {'on ' * (i + 1)}until it finally ends right here."


def _book(n_sentences: int) -> Book:
    chapter = Chapter(
        title="Only",
        paragraphs=tuple(
            Paragraph(text=_sentence(i), kind=ParagraphKind.BODY) for i in range(n_sentences)
        ),
        source_ids=("c0",),
    )
    return make_book([chapter], title="T", author="A", source_sha256=_SHA)


@dataclass
class _OverlapEngine(TTSEngine):
    """Records how many synthesize() calls were ever in flight at once."""

    name: ClassVar[str] = "overlap"
    sample_rate: int = 8000
    max_batch: int = 1
    max_concurrency: int = 3
    pause: float = 0.02
    fail_on_call: int | None = None
    # Calls that start after the failing one run slowly, so the orchestrator
    # has ample time to cancel what is still queued before workers reach it.
    pause_after_failure: float = 0.2
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    in_flight: int = 0
    high_water: int = 0
    calls: int = 0

    def synthesize(self, texts: Sequence[str]) -> list[AudioClip]:
        with self._lock:
            self.calls += 1
            call = self.calls
            self.in_flight += 1
            self.high_water = max(self.high_water, self.in_flight)
        try:
            past_failure = self.fail_on_call is not None and call > self.fail_on_call
            time.sleep(self.pause_after_failure if past_failure else self.pause)
            if call == self.fail_on_call:
                raise RuntimeError("batch exploded")
            return [
                AudioClip(samples=np.zeros(len(text), dtype=np.float32), sample_rate=8000)
                for text in texts
            ]
        finally:
            with self._lock:
                self.in_flight -= 1

    def fingerprint(self) -> str:
        return "overlap"


def test_concurrent_engine_overlaps_batches_and_keys_stay_correct(tmp_path: Path) -> None:
    book = _book(9)
    engine = _OverlapEngine()
    lines: list[str] = []
    cache_dir = tmp_path / "cache"
    (result,) = synthesize_book(
        book, engine, cache_dir=cache_dir, out_dir=tmp_path / "out", log=lines.append
    )

    assert engine.high_water >= 2
    assert len(result.cues) == 9
    for text, _start, _end in result.cues:
        key = cache.clip_cache_key(TEXT_PIPELINE_VERSION, text)
        clip = cache.load_clip(cache_dir, "overlap", key)
        assert clip is not None
        assert len(clip.samples) == len(text)
    batch_lines = [line for line in lines if line.startswith("batch")]
    assert batch_lines == [f"batch {n}/9, {n}/9 clips" for n in range(1, 10)]


def test_failing_batch_ends_the_run(tmp_path: Path) -> None:
    engine = _OverlapEngine(fail_on_call=2)
    with pytest.raises(RuntimeError, match="batch exploded"):
        synthesize_book(_book(9), engine, cache_dir=tmp_path / "cache", out_dir=tmp_path / "out")
    assert engine.in_flight == 0
    # Batches queued behind the failure were cancelled, not drained.
    assert engine.calls < 9
