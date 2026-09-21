"""Cache-aware orchestration: synthesize_book's missing/cached/stale wiring.

tests/synth/test_orchestrator.py covers the older cache-free helpers
(sentence_gap, batch_sentences, synthesize_chapter); this file covers the
M5 restructure that sits on top of them.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar

import numpy as np

from epub_to_m4b.book import AudioClip, Book, Chapter, Paragraph, ParagraphKind
from epub_to_m4b.synth.orchestrator import GapPolicy, synthesize_book
from epub_to_m4b.tts.base import TTSEngine

# Long enough that the splitter's orphan-short-fragment merge doesn't fold
# them back into one sentence (mirrors tests/synth/test_orchestrator.py).
_S1 = "This is a considerably long first sentence that should not get merged."
_S2 = "This is a second, equally long sentence that also should not get merged."


@dataclass
class _RecordingEngine(TTSEngine):
    name: ClassVar[str] = "recording"
    sample_rate: int = 8000
    max_batch: int = 8
    calls: list[list[str]] = field(default_factory=list)

    def synthesize(self, texts: Sequence[str]) -> list[AudioClip]:
        self.calls.append(list(texts))
        clips = []
        for text in texts:
            n = max(1, len(text) * 10)
            samples = np.zeros(n, dtype=np.float32)
            clips.append(AudioClip(samples=samples, sample_rate=self.sample_rate))
        return clips

    def fingerprint(self) -> str:
        return "recording:v1"

    @property
    def synthesized_texts(self) -> list[str]:
        return [text for call in self.calls for text in call]


def _book(*, n_chapters: int = 2) -> Book:
    chapters = tuple(
        Chapter(
            title=f"Chapter {i}",
            paragraphs=(
                Paragraph(text=f"Chapter {i}", kind=ParagraphKind.HEADING),
                Paragraph(text=f"{_S1} Chapter {i}: {_S2}", kind=ParagraphKind.BODY),
            ),
            source_ids=(f"c{i}",),
        )
        for i in range(n_chapters)
    )
    return Book(
        title="Test Book",
        author="Author",
        cover=None,
        cover_mime=None,
        chapters=chapters,
        source_sha256="book-sha-abc123",
    )


def test_first_run_synthesizes_every_sentence(tmp_path: Path) -> None:
    book = _book()
    engine = _RecordingEngine()
    cache_dir = tmp_path / "cache"
    out_dir = tmp_path / "out"
    results = synthesize_book(book, engine, cache_dir=cache_dir, out_dir=out_dir)

    assert len(results) == 2
    for result in results:
        assert result.flac_path.is_file()
        assert result.duration > 0
        assert len(result.cues) == 3  # heading + 2 body sentences
    total_sentences = sum(len(r.cues) for r in results)
    assert len(engine.synthesized_texts) == total_sentences


def test_second_run_with_same_book_and_dirs_synthesizes_nothing(tmp_path: Path) -> None:
    book = _book()
    cache_dir = tmp_path / "cache"
    out_dir = tmp_path / "out"
    synthesize_book(book, _RecordingEngine(), cache_dir=cache_dir, out_dir=out_dir)

    second_engine = _RecordingEngine()
    results = synthesize_book(book, second_engine, cache_dir=cache_dir, out_dir=out_dir)

    assert second_engine.calls == []  # chapters were reused wholesale
    assert len(results) == 2
    for result in results:
        assert result.flac_path.is_file()


def test_fresh_work_dir_but_same_clip_cache_reuses_clips_not_chapters(tmp_path: Path) -> None:
    # Different out_dir (so no chapter manifest exists) but the same clip
    # cache dir: chapters must be rebuilt (no manifest to reuse), but no
    # sentence should be re-synthesized - every clip is already cached.
    book = _book()
    cache_dir = tmp_path / "cache"
    synthesize_book(book, _RecordingEngine(), cache_dir=cache_dir, out_dir=tmp_path / "out1")

    second_engine = _RecordingEngine()
    results = synthesize_book(book, second_engine, cache_dir=cache_dir, out_dir=tmp_path / "out2")

    assert second_engine.calls == []
    assert len(results) == 2
    for result in results:
        assert result.flac_path.is_file()


def test_gap_policy_change_rebuilds_chapter_without_resynthesizing_clips(tmp_path: Path) -> None:
    book = _book(n_chapters=1)
    cache_dir = tmp_path / "cache"
    out_dir = tmp_path / "out"
    first = synthesize_book(
        book, _RecordingEngine(), cache_dir=cache_dir, out_dir=out_dir, policy=GapPolicy()
    )

    changed_policy = GapPolicy(heading=9.0)
    second_engine = _RecordingEngine()
    second = synthesize_book(
        book,
        second_engine,
        cache_dir=cache_dir,
        out_dir=out_dir,
        policy=changed_policy,
    )

    # the clip cache is untouched by the gap policy change...
    assert second_engine.calls == []
    # ...but the chapter's assembled duration reflects the new (much
    # longer) heading gap, proving the chapter really was rebuilt.
    assert second[0].duration > first[0].duration


def test_text_change_invalidates_only_the_affected_clip(tmp_path: Path) -> None:
    book = _book(n_chapters=1)
    cache_dir = tmp_path / "cache"
    synthesize_book(book, _RecordingEngine(), cache_dir=cache_dir, out_dir=tmp_path / "out1")

    changed_sentence = "This considerably long sentence has now changed completely, entirely."
    edited_chapter = Chapter(
        title="Chapter 0",
        paragraphs=(
            Paragraph(text="Chapter 0", kind=ParagraphKind.HEADING),
            Paragraph(text=f"{_S1} Chapter 0: {changed_sentence}", kind=ParagraphKind.BODY),
        ),
        source_ids=("c0",),
    )
    edited_book = Book(
        title=book.title,
        author=book.author,
        cover=None,
        cover_mime=None,
        chapters=(edited_chapter,),
        source_sha256=book.source_sha256,
    )
    second_engine = _RecordingEngine()
    synthesize_book(edited_book, second_engine, cache_dir=cache_dir, out_dir=tmp_path / "out2")

    # only the one sentence whose text actually changed needed synthesis.
    assert len(second_engine.synthesized_texts) == 1
    assert "changed completely" in second_engine.synthesized_texts[0]

    # the unrelated heading and first sentence stayed byte-identical, so
    # they hit the clip cache, letting the total call count stay at 1
    # regardless of how many sentences the chapter actually has.


def test_on_chapter_start_callback_reports_every_chapter(tmp_path: Path) -> None:
    book = _book()
    seen: list[tuple[int, int, str, int]] = []
    synthesize_book(
        book,
        _RecordingEngine(),
        cache_dir=tmp_path / "cache",
        out_dir=tmp_path / "out",
        on_chapter_start=lambda idx, total, title, n: seen.append((idx, total, title, n)),
    )
    assert [s[0] for s in seen] == [0, 1]
    assert all(s[1] == 2 for s in seen)


def test_max_batch_respected_across_many_sentences(tmp_path: Path) -> None:
    book = _book(n_chapters=4)
    engine = _RecordingEngine(max_batch=3)
    synthesize_book(book, engine, cache_dir=tmp_path / "cache", out_dir=tmp_path / "out")
    assert all(len(call) <= 3 for call in engine.calls)
