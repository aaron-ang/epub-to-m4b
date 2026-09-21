"""Numeric normalization: years, decades, clocks, roman numerals, ordinals.

No foreign-script romanization here - this module is English-only, so
there's no script detection to test.
"""

from __future__ import annotations

import pytest

import epub_to_m4b.text.lang.english as english_mod
from epub_to_m4b.text.lang.english import (
    clock_to_words,
    decades_to_words,
    normalize_english,
    ordinals_and_math_to_words,
    roman_numerals_to_words,
    years_to_words,
)

# ---------------------------------------------------------------------------
# years (each case needs a cue word/month/decade-suffix to fire the year
# heuristic - a bare 4-digit number stays a cardinal)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text, expected",
    [
        ("in 1900", "in nineteen hundred"),
        ("since 1905", "since nineteen oh five"),
        ("by 1960", "by nineteen sixty"),
        ("the year 2000", "the year two thousand"),
        ("circa 2005", "circa two thousand and five"),
        ("March 1997", "March nineteen ninety-seven"),
        ("in 1997 he moved", "in nineteen ninety-seven he moved"),
    ],
)
def test_years_to_words_fires_on_cue(text: str, expected: str) -> None:
    assert years_to_words(text) == expected


def test_years_to_words_skips_without_a_cue() -> None:
    # "page 1066" has no preceding cue word, no trailing 's/'s, and no
    # adjacent month name, so the year heuristic must not fire here.
    assert years_to_words("page 1066") == "page 1066"


def test_years_to_words_skips_dollar_amount() -> None:
    assert years_to_words("$2000 grant") == "$2000 grant"


def test_years_to_words_returns_input_on_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom(*_args: object, **_kwargs: object) -> str:
        raise RuntimeError("forced")

    monkeypatch.setattr(english_mod, "num2words", _boom)
    assert years_to_words("in 1900") == "in 1900"


# Documented choice: "page 1066" has no year cue, so years_to_words leaves it
# alone (see above) and numbers_to_words - later in the pipeline - reads it
# as a plain 4-digit cardinal ("one thousand and sixty-six"), not a
# two-halves year-ish "ten sixty-six" reading. A bare 4-digit number with
# no date context is a quantity, not a date, so plain cardinal is the more
# defensible default.
def test_full_pipeline_page_number_is_plain_cardinal() -> None:
    assert normalize_english("page 1066") == "page one thousand and sixty-six"


def test_full_pipeline_dollar_year_is_plain_cardinal_not_a_date() -> None:
    # Cardinal happens to look identical to the year form for a round
    # number like 2000, but it must not go through the year/decade path.
    assert normalize_english("$2000 grant") == "$two thousand grant"


# ---------------------------------------------------------------------------
# decades
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text, expected",
    [
        ("the 1960s", "the nineteen sixties"),
        ("the 1900s", "the nineteen hundreds"),
        ("the 2000s", "the two thousands"),
        ("the 1980s.", "the nineteen eighties."),
        ("the 1010s", "the ten tens"),
        ("model 1960s2", "model 1960s2"),
    ],
)
def test_decades_to_words(text: str, expected: str) -> None:
    assert decades_to_words(text) == expected


# ---------------------------------------------------------------------------
# clocks
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text, expected",
    [
        ("at 18:02:21 the", "at eighteen oh two and twenty-one seconds the"),
        ("at 18:00 the", "at eighteen hundred the"),
        ("at 13:45 the", "at thirteen forty-five the"),
        ("at 10:30:05 the", "at ten thirty and five seconds the"),
        # No 12-hour "quarter past"/"half past" idiom: without an am/pm
        # marker there's no way to know which half of the day it is, so
        # every time reads as a plain digital clock regardless of hour.
        ("at 9:15 the", "at nine fifteen the"),
        ("at 10:30 the", "at ten thirty the"),
    ],
)
def test_clock_to_words(text: str, expected: str) -> None:
    assert clock_to_words(text) == expected


def test_clock_regex_false_positive_on_a_colon_title_is_a_known_limitation() -> None:
    # "3:10 to Yuma" has the exact H:MM shape, so the regex can't tell it
    # apart from a real clock time without guessing at surrounding words -
    # accepted as a known limitation rather than over-fitting the pattern.
    assert clock_to_words("3:10 to Yuma") == "three ten to Yuma"


# ---------------------------------------------------------------------------
# roman numerals (headings only)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Chapter IV", "Chapter four"),
        ("Chapter V", "Chapter five"),
        ("Part IX", "Part nine"),
        ("See section XL for details.", "See section forty for details."),
    ],
)
def test_roman_numerals_to_words(text: str, expected: str) -> None:
    assert roman_numerals_to_words(text) == expected


def test_roman_numerals_does_not_convert_pronoun_i() -> None:
    assert roman_numerals_to_words("I think so") == "I think so"


def test_roman_numerals_does_not_convert_stray_v() -> None:
    # sanity: chapter-word context DOES convert...
    assert roman_numerals_to_words("Chapter V") == "Chapter five"
    # ...but the same letter with no such context is left alone.
    assert roman_numerals_to_words("grade V") == "grade V"


# ---------------------------------------------------------------------------
# ordinals and math
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text, expected",
    [
        ("1st place", "first place"),
        ("2nd place", "second place"),
        ("3rd place", "third place"),
        ("4th place", "fourth place"),
        ("21st place", "twenty-first place"),
        ("11th place", "eleventh place"),
    ],
)
def test_ordinals_to_words(text: str, expected: str) -> None:
    assert ordinals_and_math_to_words(text) == expected


def test_ordinal_with_mismatched_suffix_is_left_alone() -> None:
    # 21's correct suffix is "st"; "21th" doesn't match the expected-suffix
    # table, so it's treated as a typo/noise rather than an ordinal.
    assert ordinals_and_math_to_words("21th place") == "21th place"


@pytest.mark.parametrize(
    "text, expected",
    [
        ("5 + 3 = 8", "5 plus 3 equals 8"),
        ("10 - 5 = 5", "10 minus 5 equals 5"),
        ("50%", "50 percent"),
    ],
)
def test_math_operators_to_words(text: str, expected: str) -> None:
    assert ordinals_and_math_to_words(text) == expected


def test_enumeration_marker_at_line_start() -> None:
    text = "1. First item\n2. Second item"
    result = ordinals_and_math_to_words(text)
    assert result == "1 : First item\n2 : Second item"


def test_tight_numeric_range_is_not_eaten_by_math_minus() -> None:
    # "10-15" has no spaces, so ordinals_and_math_to_words' spaced-minus
    # regex must leave it alone for numbers_to_words to read as a range.
    assert ordinals_and_math_to_words("10-15") == "10-15"
    assert normalize_english("10-15") == "ten to fifteen"


# ---------------------------------------------------------------------------
# glued numbers / identifiers must never be spelled out
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "GPT-4",
        "COVID-19",
        "Boeing 747",
        "Model-1234",  # exercises the 4+-digit glued guard specifically
    ],
)
def test_identifiers_are_never_mangled(text: str) -> None:
    assert normalize_english(text) == text


# ---------------------------------------------------------------------------
# thousands-separated numbers
# ---------------------------------------------------------------------------


def test_thousands_separator_grouping() -> None:
    assert (
        normalize_english("1,234,567 people")
        == "one million, two hundred and thirty-four thousand, five hundred and sixty-seven people"
    )


def test_small_numbers_stay_as_digits_even_with_a_date_nearby() -> None:
    # "5" is under 1000 so it's left as a digit; "1997" sits next to the
    # month name "August" so the year heuristic converts it.
    assert normalize_english("August 5, 1997") == "August 5, nineteen ninety-seven"
