from __future__ import annotations

from epub_to_m4b.book import Paragraph, ParagraphKind
from epub_to_m4b.epub.chapters import (
    SpineDoc,
    TocEntry,
    build_chapters,
    flatten_toc,
    map_toc_to_spine,
    remove_running_headers,
)

H = ParagraphKind.HEADING
B = ParagraphKind.BODY
LONG = "Sentence of narration that carries the story forward. " * 6  # well over 200 chars
TITLE = "Outliers, The Story of Success"


def doc(
    id_: str,
    *paras: tuple[str, ParagraphKind] | str,
    href: str | None = None,
    types: frozenset[str] = frozenset(),
) -> SpineDoc:
    ps = [Paragraph(p, B) if isinstance(p, str) else Paragraph(*p) for p in paras]
    return SpineDoc(id_, href or f"{id_}.xhtml", ps, types)


def toc(*items: tuple[str, str] | tuple[str, str, int]) -> list[TocEntry]:
    return [TocEntry(t, h, d[0] if d else 1) for t, h, *d in items]


def titles(chapters: list[object]) -> list[str]:
    return [getattr(c, "title") for c in chapters]  # noqa: B009


# --- flatten_toc -----------------------------------------------------------


class Link:
    def __init__(self, href: str, title: str) -> None:
        self.href = href
        self.title = title


class Section:
    def __init__(self, title: str, href: str = "") -> None:
        self.title = title
        self.href = href


def test_flatten_toc_nested_depths() -> None:
    nested = [
        Link("a.xhtml", "A"),
        (Section("Part", "p.xhtml"), [Link("b.xhtml", "B"), (Section("Sub", "s.xhtml"), [])]),
    ]
    assert flatten_toc(nested) == [
        TocEntry("A", "a.xhtml", 1),
        TocEntry("Part", "p.xhtml", 1),
        TocEntry("B", "b.xhtml", 2),
        TocEntry("Sub", "s.xhtml", 2),
    ]


def test_flatten_toc_skips_hrefless_sections() -> None:
    assert flatten_toc([(Section("No link"), [Link("x.xhtml", "X")])]) == [
        TocEntry("X", "x.xhtml", 2)
    ]


# --- map_toc_to_spine -------------------------------------------------------


def test_map_strips_fragment_and_matches_basename() -> None:
    docs = [doc("a", href="text/a.xhtml"), doc("b", href="text/b.xhtml")]
    entries = toc(("A", "text/a.xhtml#frag"), ("B", "../text/b.xhtml"), ("C", "b.xhtml"))
    assert [(i, e.title) for i, e in map_toc_to_spine(entries, docs)] == [(0, "A"), (1, "B")]


def test_map_drops_unresolved_and_backwards() -> None:
    docs = [doc("a"), doc("b"), doc("c")]
    entries = toc(("C", "c.xhtml"), ("A", "a.xhtml"), ("Z", "zzz.xhtml"))
    assert [i for i, _ in map_toc_to_spine(entries, docs)] == [2]


# --- build_chapters: TOC mode ----------------------------------------------


def test_toc_mapping_spine_ranges_and_source_ids() -> None:
    docs = [doc("a", LONG), doc("a2", LONG), doc("b", LONG), doc("c", LONG)]
    chapters = build_chapters(docs, toc(("One", "a.xhtml"), ("Two", "c.xhtml")), "Book")
    assert titles(chapters) == ["One", "Two"]
    assert chapters[0].source_ids == ("a", "a2", "b")
    assert chapters[1].source_ids == ("c",)


def test_front_matter_before_toc_kept_only_when_long() -> None:
    short = [doc("front", "Title page"), doc("a", LONG)]
    assert titles(build_chapters(short, toc(("One", "a.xhtml")), "Book")) == ["One"]
    long_front = [doc("front", ("Preface", H), LONG), doc("a", LONG)]
    assert titles(build_chapters(long_front, toc(("One", "a.xhtml")), "Book")) == ["Preface", "One"]


def test_front_matter_named_contents_dropped() -> None:
    docs = [doc("front", ("Contents", H), LONG), doc("a", LONG)]
    assert titles(build_chapters(docs, toc(("One", "a.xhtml")), "Book")) == ["One"]


def test_epub_type_exclusion() -> None:
    docs = [
        doc("cover", LONG, types=frozenset({"cover"})),
        doc("copy", LONG, types=frozenset({"frontmatter", "copyright-page"})),
        doc("pref", LONG, types=frozenset({"frontmatter", "preface"})),
        doc("a", LONG, types=frozenset({"bodymatter", "chapter"})),
    ]
    entries = toc(("Cover", "cover.xhtml"), ("Preface", "pref.xhtml"), ("One", "a.xhtml"))
    chapters = build_chapters(docs, entries, "Book")
    assert titles(chapters) == ["Preface", "One"]
    assert chapters[0].source_ids == ("pref",)


def test_front_matter_labels_dropped_without_epub_types() -> None:
    docs = [doc("t", LONG), doc("c", LONG), doc("toc", LONG), doc("a", LONG)]
    entries = toc(
        ("Title Page", "t.xhtml"),
        ("Copyright", "c.xhtml"),
        ("Contents", "toc.xhtml"),
        ("One", "a.xhtml"),
    )
    assert titles(build_chapters(docs, entries, "Book")) == ["One"]


def test_note_and_source_list_labels_dropped_without_epub_types() -> None:
    docs = [doc("a", LONG), doc("n", LONG), doc("r", LONG), doc("s", LONG), doc("b", LONG)]
    entries = toc(
        ("One", "a.xhtml"),
        ("Notes", "n.xhtml"),
        ("References", "r.xhtml"),
        ("Sources", "s.xhtml"),
        ("About the Author", "b.xhtml"),
    )
    chapters = build_chapters(docs, entries, "Book")
    assert titles(chapters) == ["One", "About the Author"]
    assert [c.source_ids for c in chapters] == [("a",), ("b",)]


def test_doc_left_empty_by_note_list_semantics_excluded() -> None:
    # epub/html.py already dropped the note list; only its heading is left.
    notes = doc("n", ("Notes", H), types=frozenset({"endnotes"}))
    mixed = doc("m", ("Two", H), LONG, types=frozenset({"endnotes"}))
    chapters = build_chapters([doc("a", LONG), notes, mixed], [], "Book")
    assert titles(chapters) == ["Sentence of narration that carries the story forward.", "Two"]
    assert [c.source_ids for c in chapters] == [("a",), ("m",)]


def test_bare_number_and_truncated_labels_fold_into_previous() -> None:
    docs = [doc("a", LONG), doc("s1", LONG), doc("s2", LONG), doc("q", LONG), doc("b", LONG)]
    entries = toc(
        ("CHAPTER ONE", "a.xhtml"),
        ("1.", "s1.xhtml"),
        ("IV", "s2.xhtml"),
        ("CAVIEDES: Advise him we don't have...", "q.xhtml"),
        ("CHAPTER TWO", "b.xhtml"),
    )
    chapters = build_chapters(docs, entries, "Book")
    assert titles(chapters) == ["CHAPTER ONE", "CHAPTER TWO"]
    assert chapters[0].source_ids == ("a", "s1", "s2", "q")


def test_toc_depth_filters_entries() -> None:
    docs = [doc("p", LONG), doc("c1", LONG), doc("c2", LONG)]
    entries = toc(("Part", "p.xhtml"), ("Ch 1", "c1.xhtml", 2), ("Ch 2", "c2.xhtml", 2))
    assert titles(build_chapters(docs, entries, "Book")) == ["Part"]
    assert titles(build_chapters(docs, entries, "Book", toc_depth=2)) == ["Part", "Ch 1", "Ch 2"]


# --- heading-mode fallback --------------------------------------------------


def test_heading_mode_when_toc_covers_too_little() -> None:
    docs = [
        doc("intro", "Short blurb."),
        doc("d1", ("Chapter 1", H), LONG, ("Chapter 2", H), LONG),
        doc("d2", LONG),
        doc("d3", ("Chapter 3", H), LONG),
        doc("d4", LONG),
    ]
    chapters = build_chapters(docs, toc(("Only", "d3.xhtml")), "Book")
    assert titles(chapters) == ["Chapter 1", "Chapter 2", "Chapter 3"]
    assert chapters[1].source_ids == ("d1", "d2")
    assert chapters[2].source_ids == ("d3", "d4")


def test_heading_mode_ignores_tiny_headings() -> None:
    docs = [doc("d1", ("Chapter 1", H), LONG, ("R2", H), LONG, ("II", H), LONG)]
    assert titles(build_chapters(docs, [], "Book")) == ["Chapter 1", "II"]


def test_toc_mode_kept_at_thirty_percent_coverage() -> None:
    docs = [doc(f"d{i}", ("Heading", H), LONG) for i in range(10)]
    entries = toc(*[(f"T{i}", f"d{i}.xhtml") for i in (0, 3, 6)])
    assert titles(build_chapters(docs, entries, "Book")) == ["T0", "T3", "T6"]


# --- running headers ---------------------------------------------------------


def make_docs(title: str, n: int, with_header: bool = True) -> list[SpineDoc]:
    return [
        doc(f"d{i}", (title if with_header else f"Chapter {i}", H), f"Text {i}.") for i in range(n)
    ]


def kept_headers(docs: list[SpineDoc]) -> list[bool]:
    return [any(p.kind is H for p in d.paragraphs) for d in docs]


def test_running_header_skips_first_occurrence() -> None:
    out = remove_running_headers(make_docs(TITLE, 5), TITLE)
    assert kept_headers(out) == [True, False, False, False, False]
    assert [d.id for d in out] == ["d0", "d1", "d2", "d3", "d4"]


def test_running_header_needs_three_hits() -> None:
    assert kept_headers(remove_running_headers(make_docs(TITLE, 2), TITLE)) == [True, True]


def test_running_header_ignores_real_chapter_titles() -> None:
    out = remove_running_headers(make_docs(TITLE, 5, with_header=False), TITLE)
    assert kept_headers(out) == [True] * 5


def test_running_header_is_case_space_and_punctuation_insensitive() -> None:
    out = remove_running_headers(make_docs("OUTLIERS,  the story of success", 3), TITLE)
    assert kept_headers(out) == [True, False, False]
    out = remove_running_headers(make_docs(TITLE, 3), "Outliers the story of success")
    assert kept_headers(out) == [True, False, False]


def test_running_header_without_title() -> None:
    assert kept_headers(remove_running_headers(make_docs(TITLE, 5), None)) == [True] * 5


def test_running_header_only_checks_first_heading() -> None:
    docs = [doc(f"d{i}", ("Chapter", H), (TITLE, H), "Text.") for i in range(4)]
    assert all(
        sum(p.kind is H for p in d.paragraphs) == 2 for d in remove_running_headers(docs, TITLE)
    )


# --- stub merge and titles ---------------------------------------------------


def test_stub_merges_forward_with_prefixed_title() -> None:
    docs = [doc("p", ("PART 1", H)), doc("a", ("CHAPTER 1", H), LONG), doc("b", LONG)]
    entries = toc(("PART 1", "p.xhtml"), ("CHAPTER 1", "a.xhtml"), ("CHAPTER 2", "b.xhtml"))
    chapters = build_chapters(docs, entries, "Book")
    assert titles(chapters) == ["PART 1 — CHAPTER 1", "CHAPTER 2"]
    assert chapters[0].source_ids == ("p", "a")
    assert chapters[0].paragraphs[0].text == "PART 1"


def test_stub_titled_like_book_does_not_prefix() -> None:
    docs = [doc("t", (TITLE, H)), doc("a", LONG)]
    entries = toc((TITLE, "t.xhtml"), ("INTRODUCTION", "a.xhtml"))
    assert titles(build_chapters(docs, entries, "Outliers the story of success")) == [
        "INTRODUCTION"
    ]


def test_trailing_stub_merges_backward() -> None:
    docs = [doc("a", LONG), doc("z", "The end.")]
    chapters = build_chapters(docs, toc(("One", "a.xhtml"), ("Fin", "z.xhtml")), "Book")
    assert titles(chapters) == ["One"]
    assert chapters[0].source_ids == ("a", "z")


def test_min_chars_controls_stub_threshold() -> None:
    docs = [doc("a", LONG), doc("b", "Tiny " * 20)]
    entries = toc(("One", "a.xhtml"), ("Two", "b.xhtml"))
    assert titles(build_chapters(docs, entries, "Book", min_chars=50)) == ["One", "Two"]
    assert titles(build_chapters(docs, entries, "Book", min_chars=200)) == ["One"]


def test_title_falls_back_to_heading_then_sentence_then_number() -> None:
    docs = [
        doc("a", ("Real Heading", H), LONG),
        doc("b", "First sentence here. Second sentence."),
        doc("b2", LONG),
        doc("c", ("", B)),
        doc("c2", LONG),
    ]
    entries = toc(("", "a.xhtml"), ("", "b.xhtml"), ("", "c.xhtml"))
    got = titles(build_chapters(docs, entries, "Book"))
    assert got[:2] == ["Real Heading", "First sentence here."]
    assert got[2].startswith(("Chapter 3", "Sentence of narration"))


def test_sentence_title_truncated_to_140_bytes_with_ellipsis() -> None:
    sentence = "é" * 200 + " tail word."
    docs = [doc("a", sentence + " " + LONG)]
    (chapter,) = build_chapters(docs, toc(("", "a.xhtml")), "Book")
    assert chapter.title.endswith("…")
    assert len(chapter.title.encode()) <= 140


def test_empty_title_never_returned() -> None:
    docs = [doc("a", LONG)]
    (chapter,) = build_chapters(docs, [], "Book")
    assert chapter.title
