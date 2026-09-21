from __future__ import annotations

import hashlib
from pathlib import Path

from epub_to_m4b.book import ParagraphKind
from epub_to_m4b.epub.reader import read_book, read_spine
from tests.conftest import COVER_BYTES


def test_read_book_metadata(tiny_epub: Path) -> None:
    book = read_book(tiny_epub)
    assert book.title == "Tiny Book"
    assert book.author == "Ada Author"
    assert book.cover == COVER_BYTES
    assert book.cover_mime == "image/jpeg"
    assert book.source_sha256 == hashlib.sha256(tiny_epub.read_bytes()).hexdigest()


def test_read_book_chapters(tiny_epub: Path) -> None:
    book = read_book(tiny_epub)
    assert [c.title for c in book.chapters] == ["One: The Beginning", "Two: The End"]
    assert book.chapters[0].source_ids == ("ch1",)
    assert book.chapters[1].source_ids == ("ch2",)
    assert book.chapters[0].paragraphs[0].kind is ParagraphKind.HEADING
    assert len(book.chapters[0].paragraphs) == 3


def test_read_book_drops_short_title_page(tiny_epub: Path) -> None:
    book = read_book(tiny_epub)
    assert all("Ada Author" not in p.text for c in book.chapters for p in c.paragraphs)


def test_read_spine_order_and_hrefs(tiny_epub: Path) -> None:
    spine = read_spine(tiny_epub)
    assert [d.id for d in spine] == ["title", "ch1", "ch2"]
    assert [d.href for d in spine] == ["title.xhtml", "ch1.xhtml", "ch2.xhtml"]
    assert spine[0].paragraphs[0].text == "Tiny Book"
    assert spine[0].epub_types == frozenset()
