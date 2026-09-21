"""Plain data tables for English normalization."""

from __future__ import annotations

# Abbreviation -> full expansion. Keys keep their trailing period so the
# lookup in english.py can match "Mr." as one token instead of "Mr" + ".".
ABBREVIATIONS: dict[str, str] = {
    "Mr.": "Mister",
    "Mrs.": "Mistress",
    "Dr.": "Doctor",
    "St.": "Saint",
    "vs.": "versus",
    "etc.": "et cetera",
    "e.g.": "for example",
    "i.e.": "that is",
    "approx.": "approximately",
    "Jr.": "Junior",
    "Sr.": "Senior",
    "No.": "Number",
}

# Last digit of a number -> its ordinal suffix, for validating "21st" /
# rejecting a mismatched "21th". Anything not in this map (0, 4-9) takes "th".
ORDINAL_SUFFIXES: dict[int, str] = {
    1: "st",
    2: "nd",
    3: "rd",
}

# (old, new) literal substitutions applied in order by fix_punctuation.
# Curly/guillemet quotes -> straight quotes, en/em dash -> hyphen (collapsed
# further by a regex for runs of 2+), Unicode ellipsis -> three ASCII dots.
PUNCTUATION_PAIRS: tuple[tuple[str, str], ...] = (
    ("\u201c", '"'),  # left curly double quote
    ("\u201d", '"'),  # right curly double quote
    ("\u2018", "'"),  # left curly single quote
    ("\u2019", "'"),  # right curly single quote
    ("\u00ab", '"'),  # left guillemet
    ("\u00bb", '"'),  # right guillemet
    ("\u2013", "-"),  # en dash
    ("\u2014", "-"),  # em dash
    ("\u2026", "..."),  # horizontal ellipsis
)

# Common English month names, full and standard abbreviations, lowercase.
# Used by the year heuristic in years_to_words to detect "March 1997"-style
# dates. "may" is ambiguous with the modal verb; accepted as a known
# false-positive risk rather than special-cased out.
MONTH_NAMES: frozenset[str] = frozenset(
    {
        "january",
        "february",
        "march",
        "april",
        "may",
        "june",
        "july",
        "august",
        "september",
        "october",
        "november",
        "december",
        "jan",
        "feb",
        "mar",
        "apr",
        "jun",
        "jul",
        "aug",
        "sep",
        "sept",
        "oct",
        "nov",
        "dec",
    }
)
