from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from epub_to_m4b.audio.ffmpeg import LOUDNESS_TARGET_LUFS, probe_chapters, run_command
from epub_to_m4b.cli import main
from epub_to_m4b.epub.reader import read_book
from epub_to_m4b.synth import cache
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


def _sample_rate(m4b_path: Path) -> int:
    probe = json.loads(
        run_command(
            ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_streams", str(m4b_path)]
        )
    )
    return int(probe["streams"][0]["sample_rate"])


def _integrated_lufs(m4b_path: Path) -> float:
    # ebur128 logs its summary at info level; the last "I:" line is the
    # whole-file integrated loudness.
    result = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-nostats",
            "-i",
            str(m4b_path),
            "-vn",
            "-af",
            "ebur128",
            "-f",
            "null",
            "-",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return float(re.findall(r"I:\s+(-?[\d.]+) LUFS", result.stderr)[-1])


def test_silent_book_encodes_at_source_rate(tiny_epub: Path, tmp_path: Path) -> None:
    out_dir = tmp_path / "out"
    assert main(["convert", str(tiny_epub), "--engine", "silence", "-o", str(out_dir)]) == 0
    assert _sample_rate(out_dir / "tiny-book.m4b") == 24000


def test_tone_book_is_normalized_to_target_at_source_rate(tiny_epub: Path, tmp_path: Path) -> None:
    out_dir = tmp_path / "out"
    assert main(["convert", str(tiny_epub), "--engine", "tone", "-o", str(out_dir)]) == 0
    m4b_path = out_dir / "tiny-book.m4b"
    # loudnorm resamples internally; -ar must bring the output back
    assert _sample_rate(m4b_path) == 24000
    assert _integrated_lufs(m4b_path) == pytest.approx(LOUDNESS_TARGET_LUFS, abs=1.0)


def test_m4b_is_rebuilt_when_encode_settings_change(
    tiny_epub: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    out_dir = tmp_path / "out"
    argv = ["convert", str(tiny_epub), "--engine", "silence", "-o", str(out_dir)]
    book_sha = read_book(tiny_epub).source_sha256

    def rerun_skipped() -> bool:
        capsys.readouterr()
        assert main(argv) == 0
        return "already up to date" in capsys.readouterr().out

    assert main(argv) == 0
    first_digest = cache.load_encode_digest(out_dir, book_sha)
    assert first_digest is not None
    assert rerun_skipped()

    # No chapter flac changed; only the encode settings did.
    monkeypatch.setattr("epub_to_m4b.audio.ffmpeg.AAC_BITRATE", "128k")
    assert not rerun_skipped()
    assert cache.load_encode_digest(out_dir, book_sha) != first_digest
    assert rerun_skipped()

    monkeypatch.setattr("epub_to_m4b.audio.ffmpeg.LOUDNESS_TARGET_LUFS", -18.0)
    assert not rerun_skipped()
    assert rerun_skipped()


def test_m4b_without_encode_stamp_is_rebuilt(
    tiny_epub: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # An m4b with no encode stamp was built with unknown settings.
    out_dir = tmp_path / "out"
    argv = ["convert", str(tiny_epub), "--engine", "silence", "-o", str(out_dir)]
    assert main(argv) == 0
    cache.clear_encode_digest(out_dir, read_book(tiny_epub).source_sha256)
    capsys.readouterr()
    assert main(argv) == 0
    assert "already up to date" not in capsys.readouterr().out
