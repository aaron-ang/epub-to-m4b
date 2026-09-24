from __future__ import annotations

from epub_to_m4b.book import Book, Chapter, Paragraph, ParagraphKind
from epub_to_m4b.synth.orchestrator import (
    GapPolicy,
    book_to_sentences,
    sentence_gap,
)

_S1 = "This is a considerably long first sentence that should not get merged."
_S2 = "This is a second, equally long sentence that also should not get merged."


def _book(*chapters: Chapter) -> Book:
    return Book(
        title="B", author=None, cover=None, cover_mime=None, chapters=chapters, source_sha256="0"
    )


def test_heading_sentence_gets_heading_gap() -> None:
    chapter = Chapter(
        title="Chapter One",
        paragraphs=(
            Paragraph(text="Chapter One", kind=ParagraphKind.HEADING),
            Paragraph(text=_S1, kind=ParagraphKind.BODY),
        ),
        source_ids=("c1",),
    )
    sentences = book_to_sentences(_book(chapter))[0]
    assert sentences[0].text == "Chapter One"
    assert sentences[0].gap_after == GapPolicy().heading


def test_mid_paragraph_sentence_gets_sentence_end_gap() -> None:
    chapter = Chapter(
        title="Ch",
        paragraphs=(Paragraph(text=f"{_S1} {_S2}", kind=ParagraphKind.BODY),),
        source_ids=("c1",),
    )
    sentences = book_to_sentences(_book(chapter))[0]
    assert [s.text for s in sentences] == [_S1, _S2]
    assert sentences[0].gap_after == GapPolicy().sentence_end


def test_last_sentence_of_paragraph_not_last_of_chapter_gets_paragraph_gap() -> None:
    chapter = Chapter(
        title="Ch",
        paragraphs=(
            Paragraph(text=_S1, kind=ParagraphKind.BODY),
            Paragraph(text=_S2, kind=ParagraphKind.BODY),
        ),
        source_ids=("c1",),
    )
    sentences = book_to_sentences(_book(chapter))[0]
    assert sentences[0].text == _S1
    assert sentences[0].gap_after == GapPolicy().paragraph


def test_last_sentence_of_chapter_gets_heading_gap() -> None:
    chapter = Chapter(
        title="Ch",
        paragraphs=(
            Paragraph(text=_S1, kind=ParagraphKind.BODY),
            Paragraph(text=_S2, kind=ParagraphKind.BODY),
        ),
        source_ids=("c1",),
    )
    sentences = book_to_sentences(_book(chapter))[0]
    assert sentences[-1].text == _S2
    assert sentences[-1].gap_after == GapPolicy().heading


def test_chapter_index_is_recorded_on_every_sentence() -> None:
    chapter = Chapter(
        title="Ch", paragraphs=(Paragraph(text=_S1, kind=ParagraphKind.BODY),), source_ids=("c1",)
    )
    sentences = book_to_sentences(_book(chapter, chapter))
    assert [[s.chapter_index for s in ch] for ch in sentences] == [[0], [1]]


def test_sentence_gap_defensive_clause_case() -> None:
    policy = GapPolicy()
    gap = sentence_gap(
        "Wait;",
        ParagraphKind.BODY,
        is_last_in_paragraph=False,
        is_last_in_chapter=False,
        policy=policy,
    )
    assert gap == policy.clause


def test_sentence_gap_custom_policy_values() -> None:
    policy = GapPolicy(sentence_end=1.0, clause=2.0, paragraph=3.0, heading=4.0)
    assert (
        sentence_gap(
            "Heading",
            ParagraphKind.HEADING,
            is_last_in_paragraph=True,
            is_last_in_chapter=False,
            policy=policy,
        )
        == 4.0
    )
    assert (
        sentence_gap(
            "Body.",
            ParagraphKind.BODY,
            is_last_in_paragraph=True,
            is_last_in_chapter=False,
            policy=policy,
        )
        == 3.0
    )


def test_empty_chapter_keeps_later_chapters_aligned() -> None:
    one = Chapter(
        title="One", paragraphs=(Paragraph(text=_S1, kind=ParagraphKind.BODY),), source_ids=("c1",)
    )
    empty = Chapter(title="Empty", paragraphs=(), source_ids=("c2",))
    three = Chapter(
        title="Three",
        paragraphs=(Paragraph(text="It cost $3 in 1999.", kind=ParagraphKind.BODY),),
        source_ids=("c3",),
    )
    sentences = book_to_sentences(_book(one, empty, three))
    assert [[s.text for s in ch] for ch in sentences] == [
        [_S1],
        [],
        ["It cost three dollars in nineteen ninety nine."],
    ]
