"""XHTML document -> flat list of paragraphs."""

from __future__ import annotations

import re
import warnings
from collections.abc import Iterator

from bs4 import BeautifulSoup, NavigableString, Tag, XMLParsedAsHTMLWarning
from bs4.element import PageElement

from epub_to_m4b.book import Paragraph, ParagraphKind

HEADING_TAGS = frozenset({"h1", "h2", "h3", "h4"})
BODY_TAGS = frozenset({"p", "li", "blockquote", "dd", "dt", "figcaption"})
# Containers walked recursively; their own direct text nodes become BODY paragraphs.
CONTAINER_TAGS = frozenset(
    {"body", "div", "section", "article", "main", "header", "footer", "ul", "ol", "dl", "figure"}
)
BLOCK_TAGS = HEADING_TAGS | BODY_TAGS | CONTAINER_TAGS | {"table", "br", "hr"}
DROP_TAGS = frozenset({"script", "style", "nav", "img", "svg", "head", "title", "video", "audio"})
_NOTE_TYPES = frozenset({"footnote", "endnote", "rearnote", "note"})

# Short paragraphs that read as chapter/part labels but were marked up as plain <p>.
_NUMBER_WORDS = (
    "one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|"
    "fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty"
)
_LABEL_RE = re.compile(
    rf"^(?:chapter|part|book)\s+"
    rf"(?:\d{{1,3}}|[ivxlcdm]{{1,7}}|(?:{_NUMBER_WORDS})(?:[- ](?:{_NUMBER_WORDS}))?)"
    r"(?![a-z0-9])",
    re.IGNORECASE,
)
_LABEL_MAX_CHARS = 80


def parse_document(xhtml: bytes | str) -> tuple[list[Paragraph], frozenset[str]]:
    """Return the document's paragraphs and the epub:type tokens on body/section elements."""
    soup = _soup(xhtml)
    root: Tag = soup.body or soup
    epub_types = _epub_types(soup)
    paragraphs = [_promote_label(p) for p in _walk(root)]
    return paragraphs, epub_types


def _soup(xhtml: bytes | str) -> BeautifulSoup:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", XMLParsedAsHTMLWarning)
        try:
            return BeautifulSoup(xhtml, "lxml")
        except Exception:  # lxml rejects some malformed input; html.parser is lenient
            return BeautifulSoup(xhtml, "html.parser")


def _epub_types(soup: BeautifulSoup) -> frozenset[str]:
    found: set[str] = set()
    for el in soup.find_all(["body", "section"]):
        if isinstance(el, Tag):
            found.update(_attr(el, "epub:type").lower().split())
    return frozenset(found)


def _attr(tag: Tag, name: str) -> str:
    value = tag.get(name)
    if value is None:
        return ""
    return " ".join(value) if isinstance(value, list) else str(value)


def _promote_label(p: Paragraph) -> Paragraph:
    if p.kind is ParagraphKind.BODY and len(p.text) <= _LABEL_MAX_CHARS and _LABEL_RE.match(p.text):
        return Paragraph(p.text, ParagraphKind.HEADING)
    return p


def _collapse(text: str) -> str:
    return " ".join(text.replace("\xa0", " ").split())


def _should_drop(tag: Tag) -> bool:
    name = tag.name.lower()
    if name in DROP_TAGS:
        return True
    classes = _attr(tag, "class").lower()
    epub_type = _attr(tag, "epub:type").lower().split()
    if name in {"sup", "a"} and ("noteref" in classes or "noteref" in epub_type):
        return True
    if name == "aside" and _NOTE_TYPES.intersection(epub_type):
        return True
    return name == "figure" and not any(
        fc.get_text(strip=True) for fc in tag.find_all("figcaption")
    )


def _has_block_descendant(tag: Tag) -> bool:
    return tag.find(list(BLOCK_TAGS - {"br"})) is not None


def _walk(node: Tag) -> Iterator[Paragraph]:
    """Emit inline runs of `node` as BODY paragraphs; recurse into block children."""
    run: list[PageElement] = []

    def flush() -> Iterator[Paragraph]:
        text = _collapse(_run_text(run))
        run.clear()
        if text:
            yield Paragraph(text, ParagraphKind.BODY)

    for child in node.children:
        if not isinstance(child, Tag):
            if type(child) is NavigableString:  # skips Comment, CData, Doctype subclasses
                run.append(child)
            continue
        if _should_drop(child):
            continue
        name = child.name.lower()
        if name not in BLOCK_TAGS and not _has_block_descendant(child):
            run.append(child)
            continue
        yield from flush()
        yield from _block(child, name)
    yield from flush()


def _block(child: Tag, name: str) -> Iterator[Paragraph]:
    if name in HEADING_TAGS:
        yield from _split_on_br(child, ParagraphKind.HEADING)
    elif name == "table":
        yield from _table_rows(child)
    elif name in BODY_TAGS and not _has_block_descendant(child):
        yield from _split_on_br(child, ParagraphKind.BODY)
    elif name not in {"br", "hr"}:
        yield from _walk(child)


def _run_text(run: list[PageElement]) -> str:
    parts: list[str] = []
    for el in run:
        if isinstance(el, Tag):
            for dropped in el.find_all(_should_drop):
                dropped.decompose()
            parts.append(el.get_text(" "))
        else:
            parts.append(str(el))
    return "".join(parts)


def _split_on_br(tag: Tag, kind: ParagraphKind) -> Iterator[Paragraph]:
    """Text of a leaf block, split into separate paragraphs at each <br>."""
    run: list[PageElement] = []
    for child in tag.children:
        if isinstance(child, Tag) and child.name.lower() == "br":
            text = _collapse(_run_text(run))
            run.clear()
            if text:
                yield Paragraph(text, kind)
        elif isinstance(child, Tag):
            if not _should_drop(child):
                run.append(child)
        elif type(child) is NavigableString:
            run.append(child)
    text = _collapse(_run_text(run))
    if text:
        yield Paragraph(text, kind)


def _table_rows(table: Tag) -> Iterator[Paragraph]:
    rows = [tr for tr in table.find_all("tr") if tr.find_parent("table") is table]
    headers: list[str] = []
    for tr in rows:
        cells = [c for c in tr.find_all(["td", "th"]) if isinstance(c, Tag)]
        texts = [_collapse(c.get_text(" ", strip=True)) for c in cells]
        if cells and not headers and all(c.name.lower() == "th" for c in cells):
            headers = texts
            continue
        if headers and len(headers) == len(texts):
            texts = [f"{h}: {t}" if h else t for h, t in zip(headers, texts, strict=True) if t]
        texts = [t for t in texts if t]
        if texts:
            yield Paragraph(", ".join(texts), ParagraphKind.TABLE_ROW)
