from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path

from epub_to_m4b.book import ParagraphKind
from epub_to_m4b.epub.reader import read_book, read_spine
from epub_to_m4b.synth.orchestrator import book_to_sentences
from tests.conftest import CONTAINER_XML, COVER_BYTES
from tests.helpers import xhtml


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


_NOTES_OPF = """<?xml version="1.0" encoding="utf-8"?>
<package xmlns="http://www.idpf.org/2007/opf" unique-identifier="uid" version="2.0">
  <metadata xmlns:dc="http://purl.org/dc/elements/1.1/">
    <dc:title>Noted Book</dc:title>
    <dc:language>en</dc:language>
    <dc:identifier id="uid">urn:uuid:5678</dc:identifier>
  </metadata>
  <manifest>
    <item id="ncx" href="toc.ncx" media-type="application/x-dtbncx+xml"/>
    <item id="ch1" href="ch1.xhtml" media-type="application/xhtml+xml"/>
    <item id="notes" href="notes.xhtml" media-type="application/xhtml+xml"/>
    <item id="refs" href="refs.xhtml" media-type="application/xhtml+xml"/>
  </manifest>
  <spine toc="ncx">
    <itemref idref="ch1"/>
    <itemref idref="notes"/>
    <itemref idref="refs"/>
  </spine>
</package>
"""

_NOTES_NCX = """<?xml version="1.0" encoding="utf-8"?>
<ncx xmlns="http://www.daisy.org/z3986/2005/ncx/" version="2005-1">
  <head><meta name="dtb:uid" content="urn:uuid:5678"/></head>
  <docTitle><text>Noted Book</text></docTitle>
  <navMap>
    <navPoint id="n1" playOrder="1">
      <navLabel><text>One</text></navLabel><content src="ch1.xhtml"/>
    </navPoint>
    <navPoint id="n2" playOrder="2">
      <navLabel><text>Notes</text></navLabel><content src="notes.xhtml"/>
    </navPoint>
  </navMap>
</ncx>
"""

_BODY = " ".join(["The narrator reads this sentence aloud to the listener."] * 5)


def _build_noted_epub(path: Path) -> Path:
    docs = {
        "ch1.xhtml": xhtml(
            f"<h1>One</h1><p>{_BODY}<sup><a href='notes.xhtml#n1' id='r1'>1</a></sup></p>"
            f"<p>Wealth is assets that earn while you sleep. [78]</p>"
            f"<p>See https://fs.blog/naval-ravikant/ for more.</p>"
        ),
        "notes.xhtml": xhtml(f"<h1>Notes</h1><p id='n1'><a href='ch1.xhtml#r1'>1.</a> {_BODY}</p>"),
        "refs.xhtml": xhtml(
            f"<section epub:type='bibliography'><h1>Works</h1><p>{_BODY}</p></section>"
        ),
    }
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("mimetype", "application/epub+zip", compress_type=zipfile.ZIP_STORED)
        zf.writestr("META-INF/container.xml", CONTAINER_XML)
        zf.writestr("OEBPS/content.opf", _NOTES_OPF)
        zf.writestr("OEBPS/toc.ncx", _NOTES_NCX)
        for name, content in docs.items():
            zf.writestr(f"OEBPS/{name}", content)
    return path


def test_note_and_reference_chapters_are_excluded(tmp_path: Path) -> None:
    # notes.xhtml and refs.xhtml are dropped as chapters (structural exclusion);
    # nothing in ch1's own text is stripped: a note-ref link, a bracketed
    # marker, and a URL are all narrated as NeMo renders them.
    book = read_book(_build_noted_epub(tmp_path / "noted.epub"))
    assert [c.title for c in book.chapters] == ["One"]
    assert book.chapters[0].source_ids == ("ch1",)
    sentences = [s.text for s in book_to_sentences(book)[0]]
    assert sentences[-3:] == [
        "The narrator reads this sentence aloud to the listener.one",
        "Wealth is assets that earn while you sleep. [seventy eight]",
        "See HTTPS colon slash slash fs dot BLOG slash NAVAL-ravikant slash for more.",
    ]
