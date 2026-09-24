"""Paragraph -> sentence-sized text chunks."""

from __future__ import annotations

from epub_to_m4b.book import Paragraph, ParagraphKind
from epub_to_m4b.text.normalize import normalize
from epub_to_m4b.text.split import _sentences, split_paragraph


def _para(text: str) -> Paragraph:
    return Paragraph(text=text, kind=ParagraphKind.BODY)


def test_empty_paragraph_returns_empty_list() -> None:
    assert split_paragraph(_para("")) == []


def test_whitespace_only_paragraph_returns_empty_list() -> None:
    assert split_paragraph(_para("   \n\t  ")) == []


def test_max_chars_cap_respected_on_a_hard_cut() -> None:
    # One long word with no punctuation or space: cut mid-word at the limit.
    text = "Alphabravocharliedeltaechofoxtrotgolfhotelnospaceshere"
    pieces = split_paragraph(_para(text), max_chars=20)
    assert all(len(p) <= 20 for p in pieces)
    assert "".join(pieces) == text


def test_split_prefers_comma_over_space() -> None:
    text = "Alpha bravo charlie, delta echo foxtrot golf hotel."
    pieces = split_paragraph(_para(text), max_chars=20)
    # The cut lands right after the comma (at the 20-char limit), not at
    # an earlier or later space in the same window.
    assert pieces[0] == "Alpha bravo charlie,"


def test_comma_right_after_the_limit_is_not_kept() -> None:
    # Keeping the comma would make the first clip one char too long.
    pieces = split_paragraph(_para("a" * 20 + ", then more words"), max_chars=20)
    assert pieces[0] == "a" * 20
    assert all(len(p) <= 20 for p in pieces)


def test_split_falls_back_to_space_without_comma() -> None:
    text = "Alphabravo charliedelta echofoxtrot golfhotelindia juliet"
    pieces = split_paragraph(_para(text), max_chars=20)
    # No commas anywhere, so every cut falls on the last space before the
    # limit rather than mid-word.
    assert pieces[:4] == ["Alphabravo", "charliedelta", "echofoxtrot", "golfhotelindia"]


def test_split_hard_cut_as_last_resort() -> None:
    # No spaces or commas anywhere: nothing to cut on but raw characters.
    text = "x" * 45
    pieces = split_paragraph(_para(text), max_chars=20)
    assert pieces == ["x" * 20, "x" * 20, "x" * 5]


def test_title_is_spelled_out_before_the_split() -> None:
    text = normalize("Dr. Smith arrived home safely today.")
    expected = ["doctor Smith arrived home safely today."]
    assert split_paragraph(_para(text), max_chars=100) == expected


def test_decimal_number_does_not_split() -> None:
    text = "The value is 3.14 today."
    assert split_paragraph(_para(text), max_chars=100) == [text]


def test_quoted_closing_punctuation_stays_with_the_sentence() -> None:
    text = 'He said "stop." Then he left the room quickly.'
    # A small max_chars keeps the two sentences in separate clips, so the
    # boundary itself is visible.
    pieces = split_paragraph(_para(text), max_chars=20)
    assert pieces[0] == 'He said "stop."'


def test_consecutive_sentences_fill_a_clip_up_to_max_chars() -> None:
    # "One two. Three four." is exactly 20 chars; the next sentence would
    # push it over, so it starts the next clip.
    text = "One two. Three four. Five six seven."
    assert split_paragraph(_para(text), max_chars=20) == ["One two. Three four.", "Five six seven."]


def test_no_clip_exceeds_max_chars() -> None:
    text = (
        "Sentence one goes here without much filler at all in it now yes indeed friend. "
        "Sentence two goes here, without much filler at all in it either my friend indeed."
    )
    pieces = split_paragraph(_para(text), max_chars=60)
    assert all(len(p) <= 60 for p in pieces)
    assert " ".join(pieces) == text


# ---------------------------------------------------------------------------
# single-letter initials are not sentence ends
# ---------------------------------------------------------------------------
# Filling would rejoin a wrong split into the same string, so the
# no-split cases check the sentence boundaries before filling.


def test_name_initials_do_not_split() -> None:
    text = "He finally met J. K. Rowling at the station yesterday afternoon."
    assert _sentences(text) == [text]


def test_initial_in_place_name_does_not_split() -> None:
    text = "After many years of wandering around, they settled in S. Place for good."
    assert _sentences(text) == [text]


def test_initial_after_punctuation_does_not_split() -> None:
    text = 'The letter that arrived this morning was signed only "(A. Smith)" and nothing more.'
    assert _sentences(text) == [text]


def test_sentence_end_after_ordinary_word_still_splits() -> None:
    first = "The first sentence is long enough here."
    second = "The second one is long enough too."
    assert split_paragraph(_para(f"{first} {second}"), max_chars=40) == [first, second]


def test_sentence_end_after_acronym_still_splits() -> None:
    # "USA." ends in an uppercase letter, but not a one-letter word.
    first = "After all that, we flew back to the USA."
    second = "Then we finally went home again."
    assert split_paragraph(_para(f"{first} {second}"), max_chars=40) == [first, second]


def test_sentence_end_after_lowercase_single_letter_still_splits() -> None:
    first = "At the very end he wrote the letter x."
    second = "Then he put the pen down for good."
    assert split_paragraph(_para(f"{first} {second}"), max_chars=40) == [first, second]


def test_one_letter_sentence_final_word_is_a_known_non_split() -> None:
    # Accepted trade-off: "Plan B." reads like an initial.
    text = "When every other option had failed, we fell back on Plan B. Then it went fine."
    assert _sentences(text) == [text]


def test_title_before_initial_does_not_split() -> None:
    text = normalize(
        "Late that evening at the hospital we were introduced to Dr. J. Smith and his wife."
    )
    assert _sentences(text) == [
        "Late that evening at the hospital we were introduced to doctor J. Smith and his wife."
    ]


def test_punctuation_only_pieces_are_dropped() -> None:
    # A closing quote set off by a space, a separator line, and a lone
    # leading period have nothing to say.
    assert split_paragraph(_para("I have to land.' \"")) == ["I have to land.'"]
    assert split_paragraph(_para("* * *")) == []
    assert split_paragraph(_para("\u2193")) == []
    assert split_paragraph(_para(". A term of disparagement.")) == ["A term of disparagement."]


def test_spaced_ellipsis_leaves_no_dot_only_clip() -> None:
    text = normalize('"Hm . . . yes, all is in a man\'s hands . . . that is an axiom."')
    # Each ". " of the ellipsis ends a piece; the pieces that are only a dot
    # are dropped and the rest fills one clip.
    assert split_paragraph(_para(text)) == [
        '"Hm . yes, all is in a man\'s hands . that is an axiom."'
    ]
