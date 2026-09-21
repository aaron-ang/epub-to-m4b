from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from epub_to_m4b.audio.ffmpeg import probe_chapters
from epub_to_m4b.cli import main
from epub_to_m4b.epub.reader import read_book
from epub_to_m4b.synth.orchestrator import chapter_to_sentences

pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg/ffprobe not found on PATH",
)


def test_convert_produces_playable_m4b_and_vtt(tiny_epub: Path, tmp_path: Path) -> None:
    out_dir = tmp_path / "out"
    assert main(["convert", str(tiny_epub), "--engine", "silence", "-o", str(out_dir)]) == 0

    m4b_path = out_dir / "tiny-book.m4b"
    vtt_path = out_dir / "tiny-book.vtt"
    assert m4b_path.is_file()
    assert vtt_path.is_file()

    probe = probe_chapters(m4b_path)
    chapters = probe["chapters"]
    assert len(chapters) == 2
    assert [c["tags"]["title"] for c in chapters] == ["One: The Beginning", "Two: The End"]

    book = read_book(tiny_epub)
    expected_sentences = sum(
        len(chapter_to_sentences(chapter, idx)) for idx, chapter in enumerate(book.chapters)
    )
    vtt_text = vtt_path.read_text(encoding="utf-8")
    cue_count = vtt_text.count(" --> ")
    assert cue_count == expected_sentences
    assert cue_count > 0

    assert "[break]" not in vtt_text.lower()
    assert "[pause]" not in vtt_text.lower()
    for chapter in chapters:
        title = chapter["tags"]["title"]
        assert "[break]" not in title.lower()
        assert "[pause]" not in title.lower()
