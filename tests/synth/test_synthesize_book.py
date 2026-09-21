"""synthesize_book's resume wiring: which sentences reach the engine, which
chapters are reused wholesale, and what gets logged along the way."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar

import numpy as np
import pytest

from epub_to_m4b.book import AudioClip, Book, Chapter, Paragraph, ParagraphKind
from epub_to_m4b.synth import cache
from epub_to_m4b.synth.orchestrator import GapPolicy, synthesize_book
from epub_to_m4b.tts.base import TTSEngine

# Long enough that the splitter's orphan-short-fragment merge doesn't fold
# them back into one sentence (mirrors tests/synth/test_orchestrator.py).
_S1 = "This is a considerably long first sentence that should not get merged."
_S2 = "This is a second, equally long sentence that also should not get merged."
_SHA = "book-sha-abc123"


@dataclass
class _RecordingEngine(TTSEngine):
    name: ClassVar[str] = "recording"
    sample_rate: int = 8000
    max_batch: int = 8
    voice: str = "v1"
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
        return f"recording:{self.voice}"

    @property
    def synthesized_texts(self) -> list[str]:
        return [text for call in self.calls for text in call]


def _chapter(i: int, second_sentence: str = _S2) -> Chapter:
    return Chapter(
        title=f"Chapter {i}",
        paragraphs=(
            Paragraph(text=f"Chapter {i}", kind=ParagraphKind.HEADING),
            Paragraph(text=f"{_S1} Chapter {i}: {second_sentence}", kind=ParagraphKind.BODY),
        ),
        source_ids=(f"c{i}",),
    )


def _book(*, n_chapters: int = 2) -> Book:
    return Book(
        title="Test Book",
        author="Author",
        cover=None,
        cover_mime=None,
        chapters=tuple(_chapter(i) for i in range(n_chapters)),
        source_sha256=_SHA,
    )


def test_first_run_synthesizes_every_sentence(tmp_path: Path) -> None:
    book = _book()
    engine = _RecordingEngine()
    results = synthesize_book(book, engine, cache_dir=tmp_path / "cache", out_dir=tmp_path / "out")

    assert len(results) == 2
    for result in results:
        assert result.flac_path.is_file()
        assert result.duration > 0
        assert len(result.cues) == 3  # heading + 2 body sentences
    # _S1 is repeated verbatim in both chapters: identical text is one clip,
    # synthesized once, however many chapters it appears in.
    unique_texts = {text for r in results for text, _start, _end in r.cues}
    assert sorted(engine.synthesized_texts) == sorted(unique_texts)
    assert len(unique_texts) == 5


def test_second_run_with_same_book_and_dirs_synthesizes_nothing(tmp_path: Path) -> None:
    book = _book()
    cache_dir = tmp_path / "cache"
    out_dir = tmp_path / "out"
    first = synthesize_book(book, _RecordingEngine(), cache_dir=cache_dir, out_dir=out_dir)
    mtimes = [r.flac_path.stat().st_mtime_ns for r in first]

    second_engine = _RecordingEngine()
    second = synthesize_book(book, second_engine, cache_dir=cache_dir, out_dir=out_dir)

    assert second_engine.calls == []
    # chapters were reused wholesale, not re-assembled
    assert [r.flac_path.stat().st_mtime_ns for r in second] == mtimes
    assert [r.cues for r in second] == [r.cues for r in first]


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
    first = synthesize_book(book, _RecordingEngine(), cache_dir=cache_dir, out_dir=out_dir)

    second_engine = _RecordingEngine()
    second = synthesize_book(
        book, second_engine, cache_dir=cache_dir, out_dir=out_dir, policy=GapPolicy(heading=9.0)
    )

    # the clip cache is untouched by the gap policy change...
    assert second_engine.calls == []
    # ...but the chapter's assembled duration reflects the new (much
    # longer) heading gap, proving the chapter really was rebuilt.
    assert second[0].duration > first[0].duration


def test_switching_engine_in_the_same_out_dir_rerenders_the_chapter(tmp_path: Path) -> None:
    # Same book, same out_dir, same sentence set - but a different voice.
    # The clip cache is partitioned by fingerprint so every sentence is
    # synthesized again, and the chapter flac must be rebuilt from those new
    # clips rather than reused from the previous voice's render.
    book = _book(n_chapters=1)
    cache_dir = tmp_path / "cache"
    out_dir = tmp_path / "out"
    first = synthesize_book(book, _RecordingEngine(voice="a"), cache_dir=cache_dir, out_dir=out_dir)
    mtime_before = first[0].flac_path.stat().st_mtime_ns

    engine_b = _RecordingEngine(voice="b")
    second = synthesize_book(book, engine_b, cache_dir=cache_dir, out_dir=out_dir)

    assert len(engine_b.synthesized_texts) == len(second[0].cues)
    assert second[0].flac_path.stat().st_mtime_ns != mtime_before
    manifest = cache.load_chapter_manifest(out_dir, _SHA, 0)
    assert manifest is not None
    assert manifest.engine_fingerprint == "recording:b"


def test_text_change_invalidates_only_the_affected_clip(tmp_path: Path) -> None:
    book = _book(n_chapters=1)
    cache_dir = tmp_path / "cache"
    synthesize_book(book, _RecordingEngine(), cache_dir=cache_dir, out_dir=tmp_path / "out1")

    changed = "This considerably long sentence has now changed completely, entirely."
    edited_book = Book(
        title=book.title,
        author=book.author,
        cover=None,
        cover_mime=None,
        chapters=(_chapter(0, second_sentence=changed),),
        source_sha256=book.source_sha256,
    )
    second_engine = _RecordingEngine()
    synthesize_book(edited_book, second_engine, cache_dir=cache_dir, out_dir=tmp_path / "out2")

    # only the one sentence whose text actually changed needed synthesis;
    # the heading and first sentence stayed byte-identical and hit the cache.
    assert len(second_engine.synthesized_texts) == 1
    assert "changed completely" in second_engine.synthesized_texts[0]


def test_log_reports_cache_split_then_batches_then_assembly(tmp_path: Path) -> None:
    book = _book()
    cache_dir = tmp_path / "cache"
    out_dir = tmp_path / "out"
    lines: list[str] = []
    synthesize_book(
        book, _RecordingEngine(max_batch=4), cache_dir=cache_dir, out_dir=out_dir, log=lines.append
    )

    assert lines[:2] == [
        "[1/2] Chapter 0 — 3 sentences, 0 clips cached, 3 to synthesize",
        "[2/2] Chapter 1 — 3 sentences, 0 clips cached, 3 to synthesize",
    ]
    # _S1 repeats across chapters: 6 sentences, 5 unique clips to synthesize.
    assert lines[2:4] == ["batch 1/2, 4/5 clips", "batch 2/2, 5/5 clips"]
    assert [line.split(" (")[0] for line in lines[4:]] == [
        "[1/2] assembled Chapter 0",
        "[2/2] assembled Chapter 1",
    ]

    rerun: list[str] = []
    synthesize_book(
        book, _RecordingEngine(), cache_dir=cache_dir, out_dir=out_dir, log=rerun.append
    )
    assert rerun == [
        "[1/2] Chapter 0 — 3 sentences, chapter up to date",
        "[2/2] Chapter 1 — 3 sentences, chapter up to date",
    ]


def test_log_counts_cached_clips_when_only_the_chapter_needs_rebuilding(tmp_path: Path) -> None:
    book = _book(n_chapters=1)
    cache_dir = tmp_path / "cache"
    synthesize_book(book, _RecordingEngine(), cache_dir=cache_dir, out_dir=tmp_path / "out1")

    lines: list[str] = []
    synthesize_book(
        book, _RecordingEngine(), cache_dir=cache_dir, out_dir=tmp_path / "out2", log=lines.append
    )
    assert lines[0] == "[1/1] Chapter 0 — 3 sentences, 3 clips cached, 0 to synthesize"
    assert not any(line.startswith("batch") for line in lines)


def test_max_batch_respected_across_many_sentences(tmp_path: Path) -> None:
    book = _book(n_chapters=4)
    engine = _RecordingEngine(max_batch=3)
    synthesize_book(book, engine, cache_dir=tmp_path / "cache", out_dir=tmp_path / "out")
    assert all(len(call) <= 3 for call in engine.calls)


def test_clip_vanishing_between_synthesis_and_assembly_is_a_clear_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Assembly reloads every clip from the cache (so only one chapter's audio
    # is ever in memory); if one is gone by then, fail loudly rather than
    # assemble a chapter with a hole in it.
    monkeypatch.setattr(cache, "load_clip", lambda *_args: None)
    with pytest.raises(RuntimeError, match="vanished"):
        synthesize_book(
            _book(n_chapters=1),
            _RecordingEngine(),
            cache_dir=tmp_path / "cache",
            out_dir=tmp_path / "out",
        )
