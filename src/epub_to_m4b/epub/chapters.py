"""Pure chapter detection: TOC mapping, heading fallback, running headers, stub merge."""

from __future__ import annotations

import posixpath
import re
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any

from epub_to_m4b.book import Chapter, Paragraph, ParagraphKind

# epub:type tokens marking documents that are never read aloud.
EXCLUDED_TYPES = frozenset(
    {
        "cover",
        "toc",
        "landmark",
        "landmarks",
        "titlepage",
        "copyright-page",
        "colophon",
        "dedication",
        "acknowledgments",
        "acknowledgements",
        "glossary",
        "index",
        "bibliography",
        "frontmatter",
        "backmatter",
    }
)
# A document carrying one of these is read even if it also carries an excluded token
# (publishers often tag "frontmatter preface").
KEPT_TYPES = frozenset(
    {
        "bodymatter",
        "chapter",
        "part",
        "preface",
        "foreword",
        "introduction",
        "prologue",
        "epilogue",
        "afterword",
        "conclusion",
        "appendix",
    }
)
# TOC labels for front/back matter in books that carry no epub:type at all.
EXCLUDED_LABELS = frozenset(
    {
        "cover",
        "title page",
        "titlepage",
        "copyright",
        "copyright page",
        "copyright notice",
        "contents",
        "table of contents",
        "also by",
        "about the publisher",
        "credits",
        "newsletter sign-up",
        "begin reading",
        "dedication",
    }
)
MIN_TOC_COVERAGE = 0.30
MAX_TITLE_BYTES = 140
RUNNING_HEADER_MIN_DOCS = 3

_BARE_NUMBER_RE = re.compile(r"^(?:\d{1,3}|[ivxlcdm]{1,7})\s*[.:)]?$", re.IGNORECASE)
# Labels calibre generates from truncated paragraph text; they mark sections, not chapters.
_TRUNCATED_LABEL_RE = re.compile(r"\.\.\.$")
MIN_HEADING_KEY_CHARS = 3
_SENTENCE_END_RE = re.compile(r"(?<=[.!?])\s")
_NON_WORD_RE = re.compile(r"[^\w\s]")


@dataclass(frozen=True, slots=True)
class SpineDoc:
    id: str
    href: str
    paragraphs: list[Paragraph]
    epub_types: frozenset[str]

    @property
    def body_chars(self) -> int:
        return sum(len(p.text) for p in self.paragraphs if p.kind is not ParagraphKind.HEADING)


@dataclass(frozen=True, slots=True)
class TocEntry:
    title: str
    href: str
    depth: int


class _TitleSource(Enum):
    TOC = auto()
    HEADING = auto()
    SENTENCE = auto()
    NONE = auto()


@dataclass(slots=True)
class _Draft:
    title: str
    source: _TitleSource
    docs: list[SpineDoc]
    paragraphs: list[Paragraph]

    @property
    def body_chars(self) -> int:
        return sum(len(p.text) for p in self.paragraphs if p.kind is not ParagraphKind.HEADING)


def flatten_toc(toc: Sequence[Any], depth: int = 1) -> list[TocEntry]:
    """Flatten ebooklib's nested toc (Link | Section | (Section, [children])) in reading order."""
    out: list[TocEntry] = []
    for entry in toc:
        if isinstance(entry, tuple | list):
            head, children = entry[0], entry[1] if len(entry) > 1 else []
            out.extend(flatten_toc([head], depth))
            out.extend(flatten_toc(children, depth + 1))
            continue
        href = getattr(entry, "href", None)
        title = getattr(entry, "title", None)
        if href and title is not None:
            out.append(TocEntry(" ".join(str(title).split()), str(href), depth))
    return out


def build_chapters(
    spine: Sequence[SpineDoc],
    toc: Sequence[TocEntry],
    book_title: str,
    *,
    toc_depth: int = 1,
    min_chars: int = 200,
) -> list[Chapter]:
    docs = remove_running_headers([d for d in spine if not is_excluded_doc(d)], book_title)
    entries = [e for e in toc if e.depth <= toc_depth]
    starts = map_toc_to_spine(entries, docs)
    text_docs = [i for i, d in enumerate(docs) if d.body_chars > 0]
    covered = {i for i, _ in starts if i in text_docs}
    drafts = (
        _toc_drafts(docs, starts, min_chars)
        if text_docs and len(covered) / len(text_docs) >= MIN_TOC_COVERAGE
        else _heading_drafts(docs, min_chars)
    )
    drafts = _merge_stubs(drafts, min_chars, book_title)
    return [_finish(d, n) for n, d in enumerate(drafts, start=1)]


def is_excluded_doc(doc: SpineDoc) -> bool:
    return bool(doc.epub_types & EXCLUDED_TYPES) and not (doc.epub_types & KEPT_TYPES)


def map_toc_to_spine(
    entries: Sequence[TocEntry], docs: Sequence[SpineDoc]
) -> list[tuple[int, TocEntry]]:
    """Resolve each entry to a spine index; drops unresolvable, duplicate, or backwards targets."""
    by_href = {_norm_href(d.href): i for i, d in enumerate(docs)}
    by_base: dict[str, int] = {}
    for i, d in enumerate(docs):
        by_base.setdefault(posixpath.basename(_norm_href(d.href)), i)
    out: list[tuple[int, TocEntry]] = []
    seen: set[int] = set()
    last = -1
    for entry in entries:
        href = _norm_href(entry.href)
        idx = by_href.get(href)
        if idx is None:
            idx = by_base.get(posixpath.basename(href))
        if idx is None or idx in seen or idx < last:
            continue
        seen.add(idx)
        last = idx
        out.append((idx, entry))
    return out


def _norm_href(href: str) -> str:
    path = posixpath.normpath(href.split("#", 1)[0]) if href else ""
    while path.startswith("../"):
        path = path[3:]
    return path.lstrip("/") if path != "." else ""


def remove_running_headers(docs: Sequence[SpineDoc], title: str | None) -> list[SpineDoc]:
    """Drop the book title heading repeated at the top of many docs, keeping the first."""
    if not title:
        return list(docs)
    key = _heading_key(title)
    hits = [i for i, d in enumerate(docs) if _first_heading_key(d) == key]
    if len(hits) < RUNNING_HEADER_MIN_DOCS:
        return list(docs)
    drop = set(hits[1:])
    out: list[SpineDoc] = []
    for i, d in enumerate(docs):
        if i not in drop:
            out.append(d)
            continue
        paras = list(d.paragraphs)
        first = next(j for j, p in enumerate(paras) if p.kind is ParagraphKind.HEADING)
        del paras[first]
        out.append(SpineDoc(d.id, d.href, paras, d.epub_types))
    return out


def _heading_key(text: str) -> str:
    return " ".join(_NON_WORD_RE.sub("", text.casefold()).split())


def _first_heading_key(doc: SpineDoc) -> str | None:
    heading = next((p for p in doc.paragraphs if p.kind is ParagraphKind.HEADING), None)
    return _heading_key(heading.text) if heading else None


def _toc_drafts(
    docs: Sequence[SpineDoc], starts: Sequence[tuple[int, TocEntry]], min_chars: int
) -> list[_Draft]:
    starts = [(i, e) for n, (i, e) in enumerate(starts) if n == 0 or not _is_section_label(e.title)]
    drafts: list[_Draft] = []
    first = starts[0][0] if starts else len(docs)
    front = _draft(None, docs[:first])
    if _keep_front(front, min_chars):
        drafts.append(front)
    for n, (start, entry) in enumerate(starts):
        end = starts[n + 1][0] if n + 1 < len(starts) else len(docs)
        if entry.title.casefold() in EXCLUDED_LABELS:
            continue
        drafts.append(_draft(entry.title, docs[start:end]))
    return drafts


def _is_section_label(title: str) -> bool:
    """Bare "1." / "IV" numbers and calibre's "text..." labels are sections inside a chapter."""
    return bool(_BARE_NUMBER_RE.match(title) or _TRUNCATED_LABEL_RE.search(title))


def _keep_front(front: _Draft, min_chars: int) -> bool:
    if front.body_chars <= min_chars:
        return False
    return _title_from_paragraphs(front.paragraphs).casefold() not in EXCLUDED_LABELS


def _is_chapter_heading(p: Paragraph) -> bool:
    """Headings too short to be titles (stray superscripts, symbols) do not open chapters."""
    if p.kind is not ParagraphKind.HEADING:
        return False
    return len(_heading_key(p.text)) >= MIN_HEADING_KEY_CHARS or bool(_BARE_NUMBER_RE.match(p.text))


def _heading_drafts(docs: Sequence[SpineDoc], min_chars: int) -> list[_Draft]:
    """New chapter at every heading; text before the first heading is front matter."""
    drafts: list[_Draft] = []
    current: _Draft | None = None
    front = _Draft("", _TitleSource.NONE, [], [])
    for doc in docs:
        for p in doc.paragraphs:
            if _is_chapter_heading(p):
                current = _Draft(p.text, _TitleSource.HEADING, [doc], [p])
                drafts.append(current)
                continue
            target = current if current is not None else front
            target.paragraphs.append(p)
            if not target.docs or target.docs[-1] is not doc:
                target.docs.append(doc)
    if _keep_front(front, min_chars):
        drafts.insert(0, front)
    return drafts


def _draft(title: str | None, docs: Sequence[SpineDoc]) -> _Draft:
    paragraphs = [p for d in docs for p in d.paragraphs]
    if title:
        return _Draft(title, _TitleSource.TOC, list(docs), paragraphs)
    return _Draft("", _TitleSource.NONE, list(docs), paragraphs)


def _merge_stubs(drafts: list[_Draft], min_chars: int, book_title: str) -> list[_Draft]:
    out: list[_Draft] = []
    pending: _Draft | None = None
    for draft in drafts:
        merged = draft if pending is None else _absorb(pending, draft, book_title=book_title)
        pending = None
        if merged.body_chars < min_chars:
            pending = merged
        else:
            out.append(merged)
    if pending is not None:
        if out:
            out[-1] = _absorb(out[-1], pending, keep_first_title=True)
        else:
            out.append(pending)
    return out


def _absorb(
    head: _Draft, tail: _Draft, *, keep_first_title: bool = False, book_title: str = ""
) -> _Draft:
    """Merge `head` (earlier) into `tail`; a labelled stub prefixes the surviving title."""
    docs = head.docs + [d for d in tail.docs if d not in head.docs]
    paragraphs = head.paragraphs + tail.paragraphs
    if keep_first_title or tail.source is _TitleSource.NONE:
        return _Draft(head.title, head.source, docs, paragraphs)
    labelled = head.source in (_TitleSource.TOC, _TitleSource.HEADING) and head.title
    if labelled and _heading_key(head.title) != _heading_key(book_title):
        combined = f"{head.title} — {tail.title}"
        if len(combined.encode()) <= MAX_TITLE_BYTES:
            return _Draft(combined, tail.source, docs, paragraphs)
    return _Draft(tail.title, tail.source, docs, paragraphs)


def _finish(draft: _Draft, n: int) -> Chapter:
    title = draft.title if draft.source is not _TitleSource.NONE else ""
    if not title:
        title = _title_from_paragraphs(draft.paragraphs)
    if not title:
        title = f"Chapter {n}"
    return Chapter(
        title=title,
        paragraphs=tuple(draft.paragraphs),
        source_ids=tuple(d.id for d in draft.docs),
    )


def _title_from_paragraphs(paragraphs: Sequence[Paragraph]) -> str:
    heading = next((p for p in paragraphs if p.kind is ParagraphKind.HEADING), None)
    if heading:
        return heading.text
    body = next((p for p in paragraphs if p.kind is not ParagraphKind.HEADING), None)
    if body is None:
        return ""
    sentence = _SENTENCE_END_RE.split(body.text, maxsplit=1)[0]
    return _truncate(sentence, MAX_TITLE_BYTES)


def _truncate(text: str, max_bytes: int) -> str:
    if len(text.encode()) <= max_bytes:
        return text
    cut = text.encode()[: max_bytes - len("…".encode())].decode(errors="ignore")
    return cut.rsplit(" ", 1)[0].rstrip(" ,;:") + "…"
