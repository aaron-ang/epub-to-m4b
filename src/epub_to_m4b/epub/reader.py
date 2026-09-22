"""EPUB file -> Book, via ebooklib for the container/metadata and our own DOM parsing."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from ebooklib import ITEM_COVER, ITEM_DOCUMENT, ITEM_IMAGE, epub

from epub_to_m4b.book import Book
from epub_to_m4b.epub.chapters import (
    DEFAULT_MIN_CHARS,
    DEFAULT_TOC_DEPTH,
    SpineDoc,
    build_chapters,
    flatten_toc,
)
from epub_to_m4b.epub.html import parse_document

__all__ = ["SpineDoc", "read_book", "read_spine"]


def read_book(
    path: Path, *, toc_depth: int = DEFAULT_TOC_DEPTH, min_chars: int = DEFAULT_MIN_CHARS
) -> Book:
    ebook = _open(path)
    title = _title(ebook, path)
    spine = _spine_docs(ebook)
    chapters = build_chapters(
        spine, flatten_toc(ebook.toc), title, toc_depth=toc_depth, min_chars=min_chars
    )
    cover, mime = _cover(ebook)
    return Book(
        title=title,
        author=_author(ebook),
        cover=cover,
        cover_mime=mime,
        chapters=tuple(chapters),
        source_sha256=_sha256(path),
    )


def read_spine(path: Path) -> list[SpineDoc]:
    return _spine_docs(_open(path))


def _open(path: Path) -> Any:
    return epub.read_epub(str(path), options={"ignore_ncx": False})


def _title(ebook: Any, path: Path) -> str:
    title = " ".join(str(ebook.title or "").split())
    return title or path.stem


def _author(ebook: Any) -> str | None:
    creators = ebook.get_metadata("DC", "creator")
    for value, _attrs in creators:
        text = " ".join(str(value).split())
        if text:
            return text
    return None


def _spine_docs(ebook: Any) -> list[SpineDoc]:
    docs: list[SpineDoc] = []
    for idref, _linear in ebook.spine:
        item = ebook.get_item_with_id(idref)
        if item is None or item.get_type() != ITEM_DOCUMENT:
            continue
        paragraphs, epub_types = parse_document(item.get_content())
        docs.append(SpineDoc(str(idref), str(item.get_name()), paragraphs, epub_types))
    return docs


def _cover(ebook: Any) -> tuple[bytes | None, str | None]:
    item = next(iter(ebook.get_items_of_type(ITEM_COVER)), None)
    if item is None:
        for _value, attrs in ebook.get_metadata("OPF", "cover"):
            item = ebook.get_item_with_id(attrs.get("content", ""))
            if item is not None:
                break
    if item is None:
        for candidate in ebook.get_items_of_type(ITEM_IMAGE):
            props = " ".join(getattr(candidate, "properties", []) or [])
            if "cover" in str(candidate.get_id()).lower() or "cover-image" in props:
                item = candidate
                break
    if item is None:
        return None, None
    return bytes(item.get_content()), str(item.media_type) if item.media_type else None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()
