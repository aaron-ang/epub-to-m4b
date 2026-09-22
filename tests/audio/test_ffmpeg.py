from __future__ import annotations

from pathlib import Path

import pytest

from epub_to_m4b.audio.ffmpeg import (
    FFmpegNotFoundError,
    concat_command,
    concat_list,
    encode_m4b_command,
    ffprobe_chapters_command,
    require_ffmpeg,
    run_command,
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
    args = encode_m4b_command(audio, meta, out)
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
        "-c:a",
        "aac",
        "-b:a",
        "64k",
        "-f",
        "mp4",
        "/tmp/out.m4b",
    ]


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
    with pytest.raises(FileNotFoundError):
        run_command(["definitely-not-a-real-binary-xyz"])


def test_run_command_raises_on_nonzero_exit() -> None:
    with pytest.raises(RuntimeError, match="failed"):
        run_command(["false"])


def test_run_command_returns_stdout() -> None:
    assert run_command(["echo", "hello"]) == "hello\n"
