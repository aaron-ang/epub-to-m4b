from __future__ import annotations

from epub_to_m4b.audio.metadata import build_ffmetadata
from epub_to_m4b.book import Book, Chapter


def _book(title: str, author: str | None, chapter_titles: list[str]) -> Book:
    chapters = tuple(
        Chapter(title=t, paragraphs=(), source_ids=(f"c{i}",)) for i, t in enumerate(chapter_titles)
    )
    return Book(
        title=title,
        author=author,
        cover=None,
        cover_mime=None,
        chapters=chapters,
        source_sha256="deadbeef",
    )


def test_build_ffmetadata_basic() -> None:
    book = _book("Tiny Book", "Ada Author", ["One", "Two"])
    text = build_ffmetadata(book, [1.5, 2.0])
    assert text == (
        ";FFMETADATA1\n"
        "title=Tiny Book\n"
        "artist=Ada Author\n"
        "album=Tiny Book\n"
        "[CHAPTER]\n"
        "TIMEBASE=1/1000\n"
        "START=0\n"
        "END=1500\n"
        "title=One\n"
        "[CHAPTER]\n"
        "TIMEBASE=1/1000\n"
        "START=1500\n"
        "END=3500\n"
        "title=Two\n"
    )


def test_build_ffmetadata_no_author_omits_artist() -> None:
    book = _book("Solo Book", None, ["Only"])
    text = build_ffmetadata(book, [1.0])
    assert text == (
        ";FFMETADATA1\n"
        "title=Solo Book\n"
        "album=Solo Book\n"
        "[CHAPTER]\n"
        "TIMEBASE=1/1000\n"
        "START=0\n"
        "END=1000\n"
        "title=Only\n"
    )


def test_build_ffmetadata_escapes_special_characters_in_title() -> None:
    book = _book("Profit = Loss; #1 \\ Edition", "Author", ["Chapter One"])
    text = build_ffmetadata(book, [1.0])
    assert text.startswith(";FFMETADATA1\ntitle=Profit \\= Loss\\; \\#1 \\\\ Edition\n")


def test_build_ffmetadata_escapes_special_characters_in_chapter_title() -> None:
    book = _book("Book", "Author", ["A = B; #C"])
    text = build_ffmetadata(book, [1.0])
    assert "title=A \\= B\\; \\#C\n" in text
