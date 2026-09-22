"""Builders shared across test modules.

Plain functions rather than fixtures so a test can call them with its own
arguments inline, and so ``conftest.py`` can use them at import time.
"""

from __future__ import annotations

from collections.abc import Sequence

from epub_to_m4b.book import Book, Chapter


def xhtml(body: str) -> str:
    """Minimal XHTML document wrapping ``body``, as an EPUB content file."""
    return (
        '<?xml version="1.0" encoding="utf-8"?>\n'
        '<html xmlns="http://www.w3.org/1999/xhtml"><head><title>t</title></head>'
        f"<body>{body}</body></html>"
    )


def make_book(
    chapters: Sequence[Chapter], *, title: str, author: str | None, source_sha256: str
) -> Book:
    """A cover-less ``Book`` over ``chapters``; every other field is explicit."""
    return Book(
        title=title,
        author=author,
        cover=None,
        cover_mime=None,
        chapters=tuple(chapters),
        source_sha256=source_sha256,
    )
