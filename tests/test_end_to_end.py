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

    # Regression: stream-copying per-chapter FLACs across the concat boundary
    # left the container's own duration reflecting only the first chapter, even
    # though the chapters atom (populated independently from ffmetadata) and
    # every chapter's bytes were correct. Catch that by cross-checking the
    # container's overall duration against the last chapter's end_time.
    container_duration = float(probe["format"]["duration"])
    last_chapter_end = float(chapters[-1]["end_time"])
    assert container_duration == pytest.approx(last_chapter_end, abs=0.1)


def test_damaged_m4b_is_rebuilt_even_when_newer_than_every_chapter(
    tiny_epub: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out_dir = tmp_path / "out"
    argv = ["convert", str(tiny_epub), "--engine", "silence", "-o", str(out_dir)]
    assert main(argv) == 0
    m4b_path = out_dir / "tiny-book.m4b"

    # Simulate an encode that died partway: the file at the real path is
    # garbage, but its mtime is newer than every chapter flac, so an
    # mtime-only "is it up to date" check would keep it forever.
    with m4b_path.open("r+b") as fh:
        fh.truncate(100)
    assert main(argv) == 0
    assert m4b_path.stat().st_size > 100
    assert len(probe_chapters(m4b_path)["chapters"]) == 2
    # the encode went through a temp file that was moved into place, not
    # left behind next to the result.
    assert not [p for p in out_dir.iterdir() if p.name.startswith(".tiny-book")]

    capsys.readouterr()
    assert main(argv) == 0
    assert "already up to date" in capsys.readouterr().out
