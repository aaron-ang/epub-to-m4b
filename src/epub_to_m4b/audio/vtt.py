"""(text, start, end) cues, with offsets already accumulated across chapters, -> WEBVTT."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path


def _timestamp(seconds: float) -> str:
    total_ms = round(seconds * 1000)
    hours, rem_ms = divmod(total_ms, 3_600_000)
    minutes, rem_ms = divmod(rem_ms, 60_000)
    secs, ms = divmod(rem_ms, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{ms:03d}"


def write_vtt(cues: Sequence[tuple[str, float, float]], out_path: Path) -> None:
    lines = ["WEBVTT", ""]
    for text, start, end in cues:
        lines.append(f"{_timestamp(start)} --> {_timestamp(end)}")
        lines.append(text)
        lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")
