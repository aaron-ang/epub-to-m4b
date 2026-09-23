"""Paragraph -> sentence-sized text chunks."""

from __future__ import annotations

from epub_to_m4b.book import Paragraph, ParagraphKind
from epub_to_m4b.text.split import split_paragraph


def _para(text: str) -> Paragraph:
    return Paragraph(text=text, kind=ParagraphKind.BODY)


def test_empty_paragraph_returns_empty_list() -> None:
    assert split_paragraph(_para("")) == []


def test_whitespace_only_paragraph_returns_empty_list() -> None:
    assert split_paragraph(_para("   \n\t  ")) == []


def test_max_chars_cap_respected_when_no_merge_applies() -> None:
    # One long word-salad sentence with no punctuation: every cut piece is
    # a "real" fragment (well above the merge threshold), so none of them
    # get glued back together and the max_chars cap holds exactly.
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


def test_abbreviation_period_does_not_split() -> None:
    text = "Dr. Smith arrived home safely today."
    assert split_paragraph(_para(text), max_chars=100) == [text]


def test_decimal_number_does_not_split() -> None:
    text = "The value is 3.14 today."
    assert split_paragraph(_para(text), max_chars=100) == [text]


def test_quoted_closing_punctuation_stays_with_the_sentence() -> None:
    text = 'He said "stop." Then he left the room quickly.'
    pieces = split_paragraph(_para(text), max_chars=100)
    # Both sentences fit comfortably under max_chars=100 so they'd merge if
    # either were short enough to trigger the merge threshold; use a small
    # max_chars instead so the boundary itself is visible.
    pieces = split_paragraph(_para(text), max_chars=20)
    assert pieces[0] == 'He said "stop."'


def test_short_fragments_merge_forward() -> None:
    text = "Short. Bit. Also short. This one is much longer than the others by far."
    pieces = split_paragraph(_para(text), max_chars=20)
    # "Short." (6 chars) and "Bit." (4 chars) are each under max_chars/2=10,
    # so the first merges forward with the second rather than staying as
    # two separate near-nothing clips.
    assert pieces[0] == "Short. Bit."


def test_merge_may_exceed_max_chars_up_to_the_1_5x_ceiling() -> None:
    # A short leftover fragment from a force-cut is allowed to merge into
    # the next piece even if the combined length goes over max_chars,
    # as long as it stays within max_chars * 1.5.
    text = (
        "Sentence one goes here without much filler at all in it now yes indeed friend. "
        "Sentence two goes here without much filler at all in it either my friend indeed."
    )
    pieces = split_paragraph(_para(text), max_chars=60)
    assert all(len(p) <= 90 for p in pieces)
    assert any(len(p) > 60 for p in pieces)


# ---------------------------------------------------------------------------
# single-letter initials are not sentence ends
# ---------------------------------------------------------------------------
# Each text puts 50+ chars before the initial (max_chars=100 -> merge
# threshold 50), so a wrong split could not be hidden by _merge_short.


def test_name_initials_do_not_split() -> None:
    text = "He finally met J. K. Rowling at the station yesterday afternoon."
    assert split_paragraph(_para(text), max_chars=100) == [text]


def test_initial_in_place_name_does_not_split() -> None:
    text = "After many years of wandering around, they settled in S. Place for good."
    assert split_paragraph(_para(text), max_chars=100) == [text]


def test_initial_after_punctuation_does_not_split() -> None:
    text = 'The letter that arrived this morning was signed only "(A. Smith)" and nothing more.'
    assert split_paragraph(_para(text), max_chars=100) == [text]


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
    assert split_paragraph(_para(text), max_chars=100) == [text]


def test_abbreviation_before_initial_does_not_split() -> None:
    text = "Late that evening at the hospital we were introduced to Dr. J. Smith and his wife."
    assert split_paragraph(_para(text), max_chars=100) == [text]
