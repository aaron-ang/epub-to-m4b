"""Pure ffmpeg/ffprobe command builders, plus a thin subprocess runner.

Argument lists (not shell strings) keep every command testable without
actually running ffmpeg, and sidestep shell-quoting entirely.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from collections.abc import Sequence
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


def run_command(args: Sequence[str]) -> str:
    exe = shutil.which(args[0])
    if exe is None:
        raise FFmpegError(f"{args[0]!r} not found on PATH")
    result = subprocess.run(args, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise FFmpegError(f"{args[0]} failed (exit {result.returncode}): {result.stderr.strip()}")
    return result.stdout


def probe_chapters(m4b_path: Path) -> dict[str, Any]:
    output = run_command(ffprobe_chapters_command(m4b_path))
    return cast(dict[str, Any], json.loads(output))
