"""(text, start, end) cues, with offsets already accumulated across chapters, -> WEBVTT."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import timedelta
from pathlib import Path

_MS = timedelta(milliseconds=1)
MS_PER_SECOND = timedelta(seconds=1) // _MS
_MS_PER_MINUTE = timedelta(minutes=1) // _MS
_MS_PER_HOUR = timedelta(hours=1) // _MS


def _timestamp(seconds: float) -> str:
    total_ms = round(seconds * MS_PER_SECOND)
    hours, rem_ms = divmod(total_ms, _MS_PER_HOUR)
    minutes, rem_ms = divmod(rem_ms, _MS_PER_MINUTE)
    secs, ms = divmod(rem_ms, MS_PER_SECOND)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{ms:03d}"


def write_vtt(cues: Sequence[tuple[str, float, float]], out_path: Path) -> None:
    lines = ["WEBVTT", ""]
    for text, start, end in cues:
        lines.append(f"{_timestamp(start)} --> {_timestamp(end)}")
        lines.append(text)
        lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")
