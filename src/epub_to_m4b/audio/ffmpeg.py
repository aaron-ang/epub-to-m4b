"""Pure ffmpeg/ffprobe command builders, plus a thin subprocess runner.

Argument lists (not shell strings) keep every command testable without
actually running ffmpeg, and sidestep shell-quoting entirely.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import tempfile
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, cast

from epub_to_m4b.errors import EpubToM4bError

FFMPEG = "ffmpeg"
FFPROBE = "ffprobe"
# AAC bitrate for the mono speech track. ffmpeg's native AAC encoder caps
# its effective rate for 24 kHz mono close to this value, so a higher
# setting raises only the nominal rate, not the quality.
AAC_BITRATE = "96k"
# Integrated loudness the m4b is normalized to (EBU R128 ``loudnorm``,
# single pass; true peak and loudness range use ffmpeg's defaults).
LOUDNESS_TARGET_LUFS = -16.0


class FFmpegNotFoundError(EpubToM4bError):
    """ffmpeg and/or ffprobe is not on PATH."""


class FFmpegError(EpubToM4bError):
    """An ffmpeg/ffprobe invocation could not run or exited non-zero; the
    message carries the command name and the tool's stderr."""


def require_ffmpeg() -> None:
    missing = [name for name in (FFMPEG, FFPROBE) if shutil.which(name) is None]
    if missing:
        raise FFmpegNotFoundError(f"required on PATH but not found: {', '.join(missing)}")


def concat_list(chapter_files: Sequence[Path]) -> str:
    """ffconcat demuxer list content for stitching per-chapter files together."""
    lines = ["ffconcat version 1.0"]
    for path in chapter_files:
        escaped = str(path).replace("'", "'\\''")
        lines.append(f"file '{escaped}'")
    return "\n".join(lines) + "\n"


def concat_command(list_path: Path, output_path: Path) -> list[str]:
    # Each per-chapter FLAC restarts its own timestamp domain at 0; stream-copying
    # (`-c copy`) across that boundary leaves the concat demuxer unable to restitch
    # a single continuous timeline, so the output's STREAMINFO/duration ends up
    # reflecting only the first file even though every file's bytes are present.
    # Decoding and re-encoding through the concat instead produces one real stream.
    return [
        FFMPEG,
        "-y",
        "-loglevel",
        "error",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(list_path),
        "-c:a",
        "flac",
        str(output_path),
    ]


def encode_m4b_command(
    audio_path: Path, metadata_path: Path, output_path: Path, *, sample_rate: int
) -> list[str]:
    """Loudness-normalized AAC-in-MP4 encode. ``-ar`` pins the output to the
    source rate: loudnorm resamples internally and would otherwise hand its
    own rate on."""
    return [
        FFMPEG,
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(audio_path),
        "-i",
        str(metadata_path),
        "-map_metadata",
        "1",
        "-map",
        "0:a",
        "-af",
        f"loudnorm=I={LOUDNESS_TARGET_LUFS}",
        "-ar",
        str(sample_rate),
        "-c:a",
        "aac",
        "-b:a",
        AAC_BITRATE,
        "-f",
        "mp4",
        str(output_path),
    ]


def encode_settings_digest() -> str:
    """Digest of the encode argv with placeholder paths and sample rate, so
    any change to codec, bitrate, filter or container rebuilds the m4b."""
    template = encode_m4b_command(Path("in"), Path("meta"), Path("out"), sample_rate=0)
    return hashlib.sha256(json.dumps(template).encode("utf-8")).hexdigest()


def ffprobe_chapters_command(m4b_path: Path) -> list[str]:
    return [
        FFPROBE,
        "-v",
        "quiet",
        "-print_format",
        "json",
        "-show_format",
        "-show_chapters",
        str(m4b_path),
    ]


# ffmpeg's machine-readable progress: ``key=value`` lines on stdout, a block
# per update. ``out_time_us`` is the output position in microseconds
# (``N/A`` before the first frame). Added by the runner, not the command
# builders, so ``encode_settings_digest`` does not change.
_PROGRESS_ARGS = ("-progress", "pipe:1", "-nostats")
_OUT_TIME_KEY = "out_time_us="


def _require_executable(name: str) -> None:
    if shutil.which(name) is None:
        raise FFmpegError(f"{name!r} not found on PATH")


def run_command(args: Sequence[str]) -> str:
    _require_executable(args[0])
    result = subprocess.run(args, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise FFmpegError(f"{args[0]} failed (exit {result.returncode}): {result.stderr.strip()}")
    return result.stdout


def run_with_progress(args: Sequence[str], progress: Callable[[float], None]) -> None:
    """Run an ffmpeg command, calling ``progress`` with the output position in
    seconds at each update. stderr goes to a temp file, not a pipe, so a
    chatty ffmpeg cannot block on a full pipe while stdout is being read."""
    _require_executable(args[0])
    argv = [args[0], *_PROGRESS_ARGS, *args[1:]]
    with tempfile.TemporaryFile(mode="w+") as stderr:
        with subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=stderr, text=True) as proc:
            assert proc.stdout is not None
            for line in proc.stdout:
                if line.startswith(_OUT_TIME_KEY):
                    value = line.removeprefix(_OUT_TIME_KEY).strip()
                    if value.isdigit():
                        progress(int(value) / 1_000_000)
        if proc.returncode != 0:
            stderr.seek(0)
            raise FFmpegError(f"{args[0]} failed (exit {proc.returncode}): {stderr.read().strip()}")


def probe_chapters(m4b_path: Path) -> dict[str, Any]:
    output = run_command(ffprobe_chapters_command(m4b_path))
    return cast(dict[str, Any], json.loads(output))
