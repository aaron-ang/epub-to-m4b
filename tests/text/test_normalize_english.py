"""Abbreviations, punctuation, hyphenation, and full pipeline composition."""

from __future__ import annotations

import pytest

from epub_to_m4b.text.lang.english import (
    expand_abbreviations,
    fix_hyphenation,
    fix_punctuation,
    normalize_english,
)
from epub_to_m4b.text.normalize import LANGUAGES, normalize


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Mr. Smith met Dr. Jones.", "Mister Smith met Doctor Jones."),
        ("fruits, e.g. apples, are healthy", "fruits, for example apples, are healthy"),
        ("that is, i.e. this one", "that is, that is this one"),
        ("Saint Louis, St. Louis", "Saint Louis, Saint Louis"),
    ],
)
def test_expand_abbreviations(text: str, expected: str) -> None:
    assert expand_abbreviations(text) == expected


def test_fix_hyphenation_joins_line_wrapped_word() -> None:
    assert fix_hyphenation("self- pity") == "self-pity"


def test_fix_hyphenation_leaves_real_spaced_hyphen_alone() -> None:
    # Only a LOWERCASE continuation after the gap is treated as a wrapped
    # word; "Well - actually" is a real dash-as-punctuation usage.
    assert fix_hyphenation("Well- Actually") == "Well- Actually"


@pytest.mark.parametrize(
    "text, expected",
    [
        ("\u201cHello\u201d", '"Hello"'),
        ("it\u2019s fine", "it's fine"),
        ("Wait\u2026 really?", "Wait... really?"),
        ("a --- b", "a - b"),
    ],
)
def test_fix_punctuation(text: str, expected: str) -> None:
    assert fix_punctuation(text) == expected


def test_normalize_dispatches_through_the_registry() -> None:
    assert normalize("Mr. Smith", lang="en") == LANGUAGES["en"]("Mr. Smith")


def test_normalize_rejects_unknown_language() -> None:
    with pytest.raises(ValueError, match="fr"):
        normalize("bonjour", lang="fr")


def test_normalize_english_composition_order_years_before_numbers() -> None:
    # This is the deliberately order-sensitive case: years_to_words must run
    # before numbers_to_words. If numbers_to_words ran first, it would treat
    # the bare 4-digit "1997" as a plain cardinal ("one thousand, nine
    # hundred and ninety-seven") before years_to_words ever saw it, because
    # numbers_to_words has no year-cue awareness of its own. Running years
    # first produces the correct year reading and leaves no digits behind
    # for numbers_to_words to reprocess.
    result = normalize_english("in 1997 he moved")
    assert result == "in nineteen ninety-seven he moved"
    assert "one thousand" not in result


def test_normalize_english_composition_order_decades_before_years() -> None:
    # decades_to_words must run before years_to_words so a decade like
    # "1980s" is already spelled out (no digits left) by the time
    # years_to_words scans the text - otherwise a differently-shaped
    # regex bug could double-process the same span.
    assert normalize_english("the 1980s") == "the nineteen eighties"


def test_normalize_english_full_composition() -> None:
    text = "Mr. Smith arrived at 18:00 in 1997, self- pity in his \u201cheart\u201d."
    result = normalize_english(text)
    assert result == (
        "Mister Smith arrived at eighteen hundred in nineteen ninety-seven, "
        'self-pity in his "heart".'
    )
