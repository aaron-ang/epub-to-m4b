from __future__ import annotations

from epub_to_m4b.book import Paragraph, ParagraphKind
from epub_to_m4b.epub.html import parse_document

H = ParagraphKind.HEADING
B = ParagraphKind.BODY
R = ParagraphKind.TABLE_ROW


def paras(html: str) -> list[tuple[str, ParagraphKind]]:
    parsed, _ = parse_document(f"<html><body>{html}</body></html>")
    return [(p.text, p.kind) for p in parsed]


def test_headings_and_body() -> None:
    assert paras("<h1>Title</h1><h4>Sub</h4><p>Body text.</p><h5>Not heading</h5>") == [
        ("Title", H),
        ("Sub", H),
        ("Body text.", B),
        ("Not heading", B),
    ]


def test_br_splits_block() -> None:
    assert paras("<p>Line one<br/>Line two<br>Line three</p>") == [
        ("Line one", B),
        ("Line two", B),
        ("Line three", B),
    ]


def test_inline_tags_join_without_extra_space() -> None:
    assert paras("<p><span>Chapter</span><span> 1 HOW</span> to <i>build</i>.</p>") == [
        ("Chapter 1 HOW to build.", H),
    ]


def test_whitespace_collapsed_and_empty_skipped() -> None:
    assert paras("<p>  a \n\t b&#160;c </p><p>   </p><p></p>") == [("a b c", B)]


def test_div_with_direct_text_is_body() -> None:
    assert paras("<div>Loose text<p>Inner para</p>trailing</div>") == [
        ("Loose text", B),
        ("Inner para", B),
        ("trailing", B),
    ]


def test_div_wrapping_blocks_only_emits_blocks() -> None:
    assert paras("<div><div><p>Deep</p></div></div>") == [("Deep", B)]


def test_list_items_blockquote_dl_figcaption() -> None:
    html = (
        "<ul><li>one</li><li>two</li></ul><blockquote>quote</blockquote>"
        "<dl><dt>term</dt><dd>def</dd></dl><figure><img src='x.png'/>"
        "<figcaption>caption</figcaption></figure>"
    )
    assert [t for t, _ in paras(html)] == ["one", "two", "quote", "term", "def", "caption"]


def test_table_rows_with_header() -> None:
    html = (
        "<table><tr><th>Name</th><th>Age</th></tr>"
        "<tr><td>Ann</td><td>30</td></tr><tr><td>Bob</td><td></td></tr></table>"
    )
    assert paras(html) == [("Name: Ann, Age: 30", R), ("Name: Bob", R)]


def test_table_rows_without_header() -> None:
    assert paras("<table><tr><td>a</td><td>b</td></tr><tr><td></td></tr></table>") == [("a, b", R)]


def test_dropped_elements() -> None:
    html = (
        "<script>x=1</script><style>p{}</style><nav><p>toc</p></nav>"
        "<p>Body<sup class='noteref'>1</sup> here<sup>2</sup>.</p>"
        "<aside epub:type='footnote'><p>note</p></aside>"
        "<aside epub:type='sidebar'><p>keep</p></aside>"
        "<figure><img src='a.png'/></figure><svg><text>no</text></svg>"
    )
    assert paras(html) == [("Body here2.", B), ("keep", B)]


def test_noteref_anchor_dropped() -> None:
    assert paras("<p>Text<a epub:type='noteref' href='#n1'>1</a>.</p>") == [("Text.", B)]


def test_epub_types_extracted() -> None:
    doc = (
        "<html><body epub:type='BodyMatter chapter'>"
        "<section epub:type='chapter'><p>x</p></section></body></html>"
    )
    _, types = parse_document(doc)
    assert types == frozenset({"bodymatter", "chapter"})


def test_epub_types_absent() -> None:
    assert parse_document("<html><body><p>x</p></body></html>")[1] == frozenset()


def test_chapter_label_paragraph_promoted_to_heading() -> None:
    long_body = "Chapter 3 of the report covers " + "many things " * 10
    assert paras(
        f"<p>Chapter 12 THE EARTH MOVES</p><p>PART IV DANGEROUS</p><p>{long_body}</p>"
    ) == [
        ("Chapter 12 THE EARTH MOVES", H),
        ("PART IV DANGEROUS", H),
        (" ".join(long_body.split()), B),
    ]


def test_chapter_word_alone_stays_body() -> None:
    assert paras("<p>Chapter and verse.</p><p>Partly cloudy.</p>") == [
        ("Chapter and verse.", B),
        ("Partly cloudy.", B),
    ]


def test_bytes_input_and_xml_declaration() -> None:
    raw = b'<?xml version="1.0"?><html xmlns="http://www.w3.org/1999/xhtml"><body><p>ok</p></body></html>'
    assert parse_document(raw)[0] == [Paragraph("ok", B)]


def test_no_break_tokens_ever() -> None:
    text = " ".join(t for t, _ in paras("<p>a<br/>b</p><div>c</div><h2>d</h2>"))
    assert "[break]" not in text and "[pause]" not in text
