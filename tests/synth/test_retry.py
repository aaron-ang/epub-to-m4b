from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar

import numpy as np
import pytest

from epub_to_m4b.book import AudioClip
from epub_to_m4b.synth import cache
from epub_to_m4b.synth.batching import PendingClip
from epub_to_m4b.synth.orchestrator import synthesize_book
from epub_to_m4b.synth.retry import RetryQueue
from epub_to_m4b.tts.base import TTSEngine
from tests.synth.test_synthesize_book import _book

_RATE = 100


def _clip(seconds: float) -> AudioClip:
    return AudioClip(samples=np.zeros(round(seconds * _RATE), dtype=np.float32), sample_rate=_RATE)


@dataclass
class _ScriptedEngine(TTSEngine):
    """Clip lengths come from ``takes[text]``: index 0 is the first pass,
    index r is retry round r. A clip is good at exactly 1.0 s; its miss is
    the distance from that. A capped request stops every take at ``cap``
    seconds, and a take stopped there scores infinite."""

    name: ClassVar[str] = "scripted"
    sample_rate: int = _RATE
    max_batch: int = 3
    retries: int = 2
    takes: dict[str, list[float]] = field(default_factory=dict)
    default: list[float] = field(default_factory=lambda: [1.0])
    first_calls: list[list[str]] = field(default_factory=list)
    retry_calls: list[tuple[list[str], int, bool]] = field(default_factory=list)
    cap: float = 10.0

    def _take(self, text: str, index: int) -> float:
        lengths = self.takes.get(text, self.default)
        return lengths[min(index, len(lengths) - 1)]

    def synthesize(self, texts: Sequence[str]) -> list[AudioClip]:
        self.first_calls.append(list(texts))
        return [_clip(min(self._take(text, 0), self.cap)) for text in texts]

    def resynthesize(
        self, texts: Sequence[str], retry_round: int, *, capped: bool
    ) -> list[AudioClip]:
        self.retry_calls.append((list(texts), retry_round, capped))
        limit = self.cap if capped else math.inf
        return [_clip(min(self._take(text, retry_round), limit)) for text in texts]

    def clip_miss(self, text: str, clip: AudioClip, *, capped: bool) -> float:
        if capped and clip.seconds >= self.cap:
            return math.inf
        return abs(clip.seconds - 1.0)

    def fingerprint(self) -> str:
        return "scripted"


def _item(text: str) -> PendingClip:
    return PendingClip(key=text, text=text)


def _offer_all(queue: RetryQueue, engine: _ScriptedEngine, texts: Sequence[str]) -> list[str]:
    clips = engine.synthesize(texts)
    kept = [queue.offer(_item(t), c) for t, c in zip(texts, clips, strict=True)]
    return [s.item.text for s in kept if s is not None]


def test_good_clips_settle_at_once_and_bad_ones_queue() -> None:
    engine = _ScriptedEngine(takes={"bad": [3.0]})
    queue = RetryQueue(engine)
    assert _offer_all(queue, engine, ["ok", "bad"]) == ["ok"]
    assert len(queue) == 1


def test_no_retry_until_a_full_batch_is_queued() -> None:
    engine = _ScriptedEngine(takes={t: [3.0, 1.0] for t in ["a", "b", "c", "d"]})
    queue = RetryQueue(engine)
    _offer_all(queue, engine, ["a", "b"])
    assert queue.flush(final=False) == []
    assert engine.retry_calls == []

    _offer_all(queue, engine, ["c", "d"])
    settled = queue.flush(final=False)
    assert engine.retry_calls == [(["a", "b", "c"], 1, True)]
    assert [s.item.text for s in settled] == ["a", "b", "c"]
    assert len(queue) == 1  # "d" waits for more company


def test_final_flush_sends_the_partial_rest() -> None:
    engine = _ScriptedEngine(takes={"a": [3.0, 1.0]})
    queue = RetryQueue(engine)
    _offer_all(queue, engine, ["a"])
    settled = queue.flush(final=True)
    assert engine.retry_calls == [(["a"], 1, True)]
    assert [(s.item.text, s.clip.seconds) for s in settled] == [("a", 1.0)]
    assert settled[0].note is not None and "recovered after 1 retries" in settled[0].note
    assert len(queue) == 0


def test_still_bad_clip_requeues_under_a_new_round_until_retries_run_out() -> None:
    engine = _ScriptedEngine(max_batch=1, takes={"a": [5.0, 4.0, 1.5]})
    queue = RetryQueue(engine)
    _offer_all(queue, engine, ["a"])
    settled = queue.flush(final=True)
    assert engine.retry_calls == [(["a"], 1, True), (["a"], 2, True)]
    assert [s.clip.seconds for s in settled] == [1.5]
    assert settled[0].note is not None
    assert "kept at 1.5s, 0.5s outside" in settled[0].note
    assert "after 2 retries" in settled[0].note


def test_best_take_wins_not_the_latest_and_nothing_is_cut() -> None:
    engine = _ScriptedEngine(max_batch=1, takes={"a": [9.0, 1.4, 7.0]})
    queue = RetryQueue(engine)
    _offer_all(queue, engine, ["a"])
    (settled,) = queue.flush(final=True)
    assert settled.clip.seconds == 1.4


def test_first_pass_kept_when_every_retry_is_worse() -> None:
    engine = _ScriptedEngine(max_batch=1, takes={"a": [1.2, 3.0, 4.0]})
    queue = RetryQueue(engine)
    _offer_all(queue, engine, ["a"])
    (settled,) = queue.flush(final=True)
    assert settled.clip.seconds == 1.2


def test_take_stopped_by_the_cap_loses_to_any_complete_take() -> None:
    engine = _ScriptedEngine(max_batch=1, cap=5.0, takes={"a": [50.0, 4.0, 50.0]})
    queue = RetryQueue(engine)
    _offer_all(queue, engine, ["a"])
    (settled,) = queue.flush(final=True)
    assert settled.clip.seconds == 4.0
    assert [capped for _texts, _round, capped in engine.retry_calls] == [True, True]


def test_every_take_capped_gets_one_last_uncapped_round() -> None:
    engine = _ScriptedEngine(max_batch=2, cap=5.0, takes={"a": [50.0, 60.0, 70.0, 30.0]})
    queue = RetryQueue(engine)
    _offer_all(queue, engine, ["a", "ok"])
    (settled,) = queue.flush(final=True)
    assert engine.retry_calls == [(["a"], 1, True), (["a"], 2, True), (["a"], 3, False)]
    assert settled.clip.seconds == 30.0  # complete, never cut
    assert settled.note is not None and "after 3 retries" in settled.note


def test_uncapped_round_waits_for_a_full_batch_until_final() -> None:
    engine = _ScriptedEngine(max_batch=2, retries=1, cap=5.0, default=[50.0, 50.0, 1.0])
    queue = RetryQueue(engine)
    _offer_all(queue, engine, ["a", "b"])
    queue.flush(final=False)  # capped round: both still stopped by the cap
    assert engine.retry_calls == [(["a", "b"], 1, True), (["a", "b"], 2, False)]
    assert len(queue) == 0

    _offer_all(queue, engine, ["c"])
    assert queue.flush(final=False) == []
    assert len(engine.retry_calls) == 2


def test_engine_without_retries_settles_everything() -> None:
    engine = _ScriptedEngine(retries=0, takes={"a": [9.0]})
    queue = RetryQueue(engine)
    assert _offer_all(queue, engine, ["a"]) == ["a"]
    assert queue.flush(final=True) == []


def test_wrong_retry_clip_count_raises() -> None:
    class _Short(_ScriptedEngine):
        def resynthesize(
            self, texts: Sequence[str], retry_round: int, *, capped: bool
        ) -> list[AudioClip]:
            return []

    engine = _Short(takes={"a": [3.0]})
    queue = RetryQueue(engine)
    _offer_all(queue, engine, ["a"])
    with pytest.raises(ValueError, match="retry clips"):
        queue.flush(final=True)


def test_synthesize_book_batches_retries_and_stores_the_best_take(tmp_path: Path) -> None:
    book = _book(n_chapters=2)
    # Every text runs away on the first pass and is fine on the first retry.
    engine = _ScriptedEngine(max_batch=2, default=[3.0, 1.0])
    lines: list[str] = []
    synthesize_book(
        book, engine, cache_dir=tmp_path / "cache", out_dir=tmp_path / "out", log=lines.append
    )

    first = [t for call in engine.first_calls for t in call]
    retried = [t for call, _round, _capped in engine.retry_calls for t in call]
    assert sorted(retried) == sorted(first)
    assert all(len(call) == 2 for call, _round, _capped in engine.retry_calls[:-1])
    assert [r for _call, r, _capped in engine.retry_calls] == list(
        range(1, len(engine.retry_calls) + 1)
    )
    assert all(capped for _call, _round, capped in engine.retry_calls)
    for text in first:
        stored = cache.load_clip(tmp_path / "cache", "scripted", cache.clip_cache_key(text))
        assert stored is not None and stored.seconds == pytest.approx(1.0)
    assert any("queued for retry" in line for line in lines)
