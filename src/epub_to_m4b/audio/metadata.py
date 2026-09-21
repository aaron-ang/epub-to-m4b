"""ffmetadata text (title/artist/album/chapters) and cover embedding.

ffmpeg's own cover embedding in an m4b container is unreliable, so the
cover goes in with mutagen after the m4b is already muxed.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from mutagen.mp4 import MP4, MP4Cover

from epub_to_m4b.book import Book

# ffmetadata escaping rule: these five characters must be backslash-escaped
# wherever they appear in a key or value.
_ESCAPE_CHARS = "=;#\\\n"


def _escape(value: str) -> str:
    out: list[str] = []
    for ch in value:
        if ch in _ESCAPE_CHARS:
            out.append("\\")
        out.append(ch)
    return "".join(out)


def build_ffmetadata(book: Book, chapter_durations: Sequence[float]) -> str:
    lines = [";FFMETADATA1", f"title={_escape(book.title)}"]
    if book.author:
        lines.append(f"artist={_escape(book.author)}")
    lines.append(f"album={_escape(book.title)}")

    cursor_ms = 0
    for chapter, duration in zip(book.chapters, chapter_durations, strict=True):
        start_ms = cursor_ms
        end_ms = cursor_ms + round(duration * 1000)
        lines += [
            "[CHAPTER]",
            "TIMEBASE=1/1000",
            f"START={start_ms}",
            f"END={end_ms}",
            f"title={_escape(chapter.title)}",
        ]
        cursor_ms = end_ms
    return "\n".join(lines) + "\n"


def embed_cover(m4b_path: Path, cover: bytes, cover_mime: str) -> None:
    audio = MP4(m4b_path)
    image_format = MP4Cover.FORMAT_PNG if "png" in cover_mime else MP4Cover.FORMAT_JPEG
    audio["covr"] = [MP4Cover(cover, imageformat=image_format)]
    audio.save()
