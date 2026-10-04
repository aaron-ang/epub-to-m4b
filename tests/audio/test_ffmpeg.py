from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from epub_to_m4b.audio.ffmpeg import (
    FFmpegError,
    FFmpegNotFoundError,
    concat_command,
    concat_list,
    encode_m4b_command,
    encode_settings_digest,
    ffprobe_chapters_command,
    require_ffmpeg,
    run_command,
    run_with_progress,
)


def test_concat_list_format() -> None:
    files = [Path("/tmp/0000.flac"), Path("/tmp/0001.flac")]
    assert concat_list(files) == (
        "ffconcat version 1.0\nfile '/tmp/0000.flac'\nfile '/tmp/0001.flac'\n"
    )


def test_concat_list_escapes_single_quotes() -> None:
    files = [Path("/tmp/it's a chapter.flac")]
    assert concat_list(files) == "ffconcat version 1.0\nfile '/tmp/it'\\''s a chapter.flac'\n"


def test_concat_command_args() -> None:
    # Must decode+re-encode (not `-c copy`): each per-chapter FLAC restarts its own
    # timestamp domain at 0, and stream-copying across that boundary leaves the
    # concat demuxer unable to restitch a single continuous timeline - the output's
    # duration silently ends up reflecting only the first file.
    args = concat_command(Path("/tmp/list.txt"), Path("/tmp/out.flac"))
    assert args == [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        "/tmp/list.txt",
        "-c:a",
        "flac",
        "/tmp/out.flac",
    ]
    assert "copy" not in args


def test_encode_m4b_command_args() -> None:
    audio, meta, out = Path("/tmp/combined.flac"), Path("/tmp/meta.txt"), Path("/tmp/out.m4b")
    args = encode_m4b_command(audio, meta, out, sample_rate=24000)
    assert args == [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-i",
        "/tmp/combined.flac",
        "-i",
        "/tmp/meta.txt",
        "-map_metadata",
        "1",
        "-map",
        "0:a",
        "-af",
        "loudnorm=I=-16.0",
        "-ar",
        "24000",
        "-c:a",
        "aac",
        "-b:a",
        "96k",
        "-f",
        "mp4",
        "/tmp/out.m4b",
    ]


def test_encode_settings_digest_tracks_encode_args(monkeypatch: pytest.MonkeyPatch) -> None:
    base = encode_settings_digest()
    assert base == encode_settings_digest()
    monkeypatch.setattr("epub_to_m4b.audio.ffmpeg.AAC_BITRATE", "128k")
    assert encode_settings_digest() != base
    monkeypatch.undo()
    monkeypatch.setattr("epub_to_m4b.audio.ffmpeg.LOUDNESS_TARGET_LUFS", -18.0)
    assert encode_settings_digest() != base


def test_ffprobe_chapters_command_args() -> None:
    args = ffprobe_chapters_command(Path("/tmp/out.m4b"))
    assert args == [
        "ffprobe",
        "-v",
        "quiet",
        "-print_format",
        "json",
        "-show_format",
        "-show_chapters",
        "/tmp/out.m4b",
    ]


def test_require_ffmpeg_raises_when_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("shutil.which", lambda _name: None)
    with pytest.raises(FFmpegNotFoundError, match="ffmpeg"):
        require_ffmpeg()


def test_run_command_raises_for_missing_executable() -> None:
    with pytest.raises(FFmpegError, match="not found on PATH"):
        run_command(["definitely-not-a-real-binary-xyz"])


def test_run_command_raises_on_nonzero_exit(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(args: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(args, 1, stdout="", stderr="Invalid data found\n")

    monkeypatch.setattr(subprocess, "run", fake_run)
    with pytest.raises(FFmpegError, match=r"ffmpeg failed \(exit 1\): Invalid data found"):
        run_command(["ffmpeg", "-i", "in.flac", "out.m4b"])


def test_run_command_returns_stdout() -> None:
    assert run_command(["echo", "hello"]) == "hello\n"


_FAKE_FFMPEG = """#!/bin/sh
echo "$@" > "$(dirname "$0")/argv.txt"
echo out_time_us=N/A
echo out_time_us=1500000
echo progress=continue
echo out_time_us=3000000
echo progress=end
echo boom >&2
exit {code}
"""


def _fake_ffmpeg(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, code: int) -> Path:
    exe = tmp_path / "ffmpeg"
    exe.write_text(_FAKE_FFMPEG.format(code=code), encoding="utf-8")
    exe.chmod(0o755)
    monkeypatch.setenv("PATH", f"{tmp_path}:{os.environ['PATH']}")
    return tmp_path / "argv.txt"


def test_run_with_progress_reports_output_seconds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    argv_path = _fake_ffmpeg(tmp_path, monkeypatch, code=0)
    seen: list[float] = []
    run_with_progress(["ffmpeg", "-i", "in.flac", "out.m4b"], seen.append)
    assert seen == [1.5, 3.0]
    assert argv_path.read_text(encoding="utf-8").split() == [
        "-progress",
        "pipe:1",
        "-nostats",
        "-i",
        "in.flac",
        "out.m4b",
    ]


def test_run_with_progress_raises_on_nonzero_exit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fake_ffmpeg(tmp_path, monkeypatch, code=1)
    with pytest.raises(FFmpegError, match=r"ffmpeg failed \(exit 1\): boom"):
        run_with_progress(["ffmpeg", "-i", "in.flac", "out.m4b"], lambda _s: None)


def test_run_with_progress_raises_for_missing_executable() -> None:
    with pytest.raises(FFmpegError, match="not found on PATH"):
        run_with_progress(["definitely-not-a-real-binary-xyz"], lambda _s: None)
