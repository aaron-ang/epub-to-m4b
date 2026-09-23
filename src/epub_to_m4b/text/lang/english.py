"""English text normalization: numbers, dates, clocks, abbreviations.

A 4-digit number reads as a year only when a regex heuristic finds a cue
(a preceding "in"/"since"/month name, a following "'s", or similar) —
no NER model, no GPU, just cheap context checks around the number itself.
"""

from __future__ import annotations

import re

from num2words import num2words

from epub_to_m4b.text.lang.tables_en import (
    ABBREV_RE,
    ABBREVIATIONS,
    MONTH_NAMES,
    ORDINAL_SUFFIXES,
    PUNCTUATION_PAIRS,
)

# Chars of surrounding text checked for year-cue words (e.g. "in", "since",
# month names) before/after a candidate 4-digit year.
_YEAR_CONTEXT_CHARS = 24

# ---------------------------------------------------------------------------
# decades: "1960s" -> "nineteen sixties"
# ---------------------------------------------------------------------------

# Exactly the \d{3}0s shape (e.g. "1960s", "2000s", "1010s"); the word
# boundary at both ends rejects "model 1960s2" and mid-word digit runs.
_DECADE_RE = re.compile(r"\b(\d{2})(\d)0s\b")


def _decade_repl(match: re.Match[str]) -> str:
    first_two = int(match.group(1))
    tens = int(match.group(2))
    if first_two == 20 and tens == 0:
        return "two thousands"
    head = num2words(first_two)
    if tens == 0:
        return f"{head} hundreds"
    if tens == 1:
        return f"{head} tens"
    tail = num2words(tens * 10)
    return f"{head} {tail[:-1]}ies"


def decades_to_words(text: str) -> str:
    return _DECADE_RE.sub(_decade_repl, text)


# ---------------------------------------------------------------------------
# years: "in 1997" -> "in nineteen ninety-seven"
# ---------------------------------------------------------------------------

# Candidate 4-digit years, 1000-2099. Requires a word boundary on both sides,
# so it never matches a number already consumed by decades_to_words (a
# trailing "s" leaves no boundary right after the digits, e.g. "1960s").
_YEAR_RE = re.compile(r"\b(1\d{3}|20\d{2})\b")

# Words that, immediately before a candidate year, mark it as a date rather
# than a plain count/page-number/dollar-amount.
_YEAR_CUE_WORDS = frozenset(
    {"in", "since", "by", "of", "around", "circa", "from", "until", "to", "year"}
)

_WORD_RE = re.compile(r"[A-Za-z']+")  # a run of letters, for tokenizing context around a match


def _year_context_ok(text: str, start: int, end: int) -> bool:
    before = text[max(0, start - _YEAR_CONTEXT_CHARS) : start]
    after = text[end : end + _YEAR_CONTEXT_CHARS]
    before_words = _WORD_RE.findall(before)
    if before_words:
        last = before_words[-1].lower()
        if last in _YEAR_CUE_WORDS:
            return True
        if len(before_words) >= 2 and before_words[-2].lower() == "the" and last == "year":
            return True
    # Followed directly by "'s" or a bare "s" (not itself the start of a
    # longer word) reads as a date/decade-ish reference: "1969's", "1997s".
    if after.startswith("'s"):
        return True
    if after and after[0] == "s" and (len(after) == 1 or not after[1].isalnum()):
        return True
    # Adjacent (within 2 tokens on either side) to a month name: "March 1997".
    after_words = _WORD_RE.findall(after)
    for candidate in (*before_words[-2:], *after_words[:2]):
        if candidate.lower() in MONTH_NAMES:
            return True
    return False


def _year_word(year: int) -> str:
    first_two, last_two = divmod(year, 100)
    if first_two == 20:
        return str(num2words(year))
    if last_two == 0:
        return f"{num2words(first_two)} hundred"
    if last_two < 10:
        return f"{num2words(first_two)} oh {num2words(last_two)}"
    return f"{num2words(first_two)} {num2words(last_two)}"


def years_to_words(text: str) -> str:
    def repl(match: re.Match[str]) -> str:
        if not _year_context_ok(text, match.start(), match.end()):
            return match.group(0)
        return _year_word(int(match.group(0)))

    return _YEAR_RE.sub(repl, text)


# ---------------------------------------------------------------------------
# roman numerals: "Chapter IV" -> "Chapter four" (headings only)
# ---------------------------------------------------------------------------

_ROMAN_VALUES = (
    ("M", 1000),
    ("CM", 900),
    ("D", 500),
    ("CD", 400),
    ("C", 100),
    ("XC", 90),
    ("L", 50),
    ("XL", 40),
    ("X", 10),
    ("IX", 9),
    ("V", 5),
    ("IV", 4),
    ("I", 1),
)

# The single-letter numerals, e.g. "MDCLXVI" - the character class every
# roman-token regex below matches.
_ROMAN_LETTERS = "".join(token for token, _ in _ROMAN_VALUES if len(token) == 1)

# "Chapter"/"Part"/"Book" followed by a roman token, including single letters
# (I/V/X) which are only converted in this enumerative context.
_ROMAN_AFTER_CHAPTER_RE = re.compile(rf"\b((?i:chapter|part|book))\s+([{_ROMAN_LETTERS}]{{1,9}})\b")

# A bare multi-letter roman token elsewhere (e.g. a heading "IV. The Storm").
# Single letters are excluded here on purpose: a lone "I" is almost always
# the pronoun and a lone "V"/"X" is almost always a stray letter, not a
# numeral.
_ROMAN_STANDALONE_RE = re.compile(rf"(?<!\w)([{_ROMAN_LETTERS}]{{2,9}})(?!\w)")

# Validates the candidate is a real roman numeral (rejects junk like "MMMM"
# or "IIII" that happens to be made of roman letters but isn't a legal
# numeral).
_ROMAN_VALID_RE = re.compile(r"^(?=.)M{0,3}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})$")


def _roman_to_int(roman: str) -> int:
    roman = roman.upper()
    total = 0
    i = 0
    while i < len(roman):
        for token, value in _ROMAN_VALUES:
            if roman.startswith(token, i):
                total += value
                i += len(token)
                break
        else:  # pragma: no cover - unreachable once _ROMAN_VALID_RE has passed
            raise ValueError(f"not a roman numeral: {roman!r}")
    return total


def roman_numerals_to_words(text: str) -> str:
    def repl_chapter(match: re.Match[str]) -> str:
        word, roman = match.group(1), match.group(2)
        if not _ROMAN_VALID_RE.fullmatch(roman):
            return match.group(0)
        return f"{word} {num2words(_roman_to_int(roman))}"

    def repl_standalone(match: re.Match[str]) -> str:
        roman = match.group(1)
        if not _ROMAN_VALID_RE.fullmatch(roman):
            return match.group(0)
        return str(num2words(_roman_to_int(roman)))

    text = _ROMAN_AFTER_CHAPTER_RE.sub(repl_chapter, text)
    return _ROMAN_STANDALONE_RE.sub(repl_standalone, text)


# ---------------------------------------------------------------------------
# clocks: "18:00" -> "eighteen hundred"
# ---------------------------------------------------------------------------

# H:MM or H:MM:SS, 1-2 digit hour, exactly 2 digit minute/second. Word
# boundaries alone reject the common false positives in practice: decimals
# don't contain ':', and a version-like "3.10" doesn't match either. A real
# false positive this does NOT filter is a colon-separated title like
# "3:10 to Yuma" - accepted as a known limitation rather than over-fitting
# the regex to guess intent from surrounding words.
_CLOCK_RE = re.compile(r"\b(\d{1,2}):([0-5]\d)(?::([0-5]\d))?\b")


def _clock_repl(match: re.Match[str]) -> str:
    hour = int(match.group(1))
    minute = int(match.group(2))
    second_group = match.group(3)
    second = int(second_group) if second_group is not None else None
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        return match.group(0)
    hour_word = num2words(hour)
    if minute == 0:
        phrase = f"{hour_word} hundred"
    elif minute < 10:
        phrase = f"{hour_word} oh {num2words(minute)}"
    else:
        phrase = f"{hour_word} {num2words(minute)}"
    if second is not None and second > 0:
        phrase = f"{phrase} and {num2words(second)} seconds"
    return phrase


def clock_to_words(text: str) -> str:
    return _CLOCK_RE.sub(_clock_repl, text)


# ---------------------------------------------------------------------------
# ordinals, standalone math operators, and numbered-list markers
# ---------------------------------------------------------------------------

# "1." / "1)" at the start of a line: a numbered-list marker, kept as a
# digit + colon (matches how audiobook narrators read numbered lists; the
# number itself is not spelled out here).
_ENUM_LINE_RE = re.compile(r"^(\d+)[.)]\s+", re.MULTILINE)

# 1st/2nd/3rd/4th..., not glued into a larger word/number.
_ORDINAL_RE = re.compile(r"\b(\d+)(st|nd|rd|th)\b")

# Arithmetic minus REQUIRES surrounding spaces ("5 - 3") so it never eats a
# tight numeric range like "10-15", which numbers_to_words turns into
# "ten to fifteen" instead.
_MATH_MINUS_RE = re.compile(r"(?<=\d)\s+-\s+(?=\d)")
_MATH_PLUS_RE = re.compile(r"(?<=\d)\s*\+\s*(?=\d)")  # "+" has no other meaning between numbers
_MATH_EQUALS_RE = re.compile(r"(?<=\d)\s*=\s*(?=\d)")  # "=" has no other meaning between numbers
_MATH_PERCENT_RE = re.compile(r"(?<=\d)%")  # a "%" directly after a number


def _expected_ordinal_suffix(n: int) -> str:
    last_two = n % 100
    if 11 <= last_two <= 13:
        return "th"
    return ORDINAL_SUFFIXES.get(n % 10, "th")


def _ordinal_repl(match: re.Match[str]) -> str:
    n = int(match.group(1))
    suffix = match.group(2)
    if suffix != _expected_ordinal_suffix(n):
        return match.group(0)
    try:
        return str(num2words(n, to="ordinal"))
    except OverflowError:
        # num2words has no name for magnitudes past its largest scale word;
        # a digit run that long is not prose, leave it verbatim.
        return match.group(0)


def ordinals_and_math_to_words(text: str) -> str:
    text = _ENUM_LINE_RE.sub(r"\1 : ", text)
    text = _ORDINAL_RE.sub(_ordinal_repl, text)
    text = _MATH_MINUS_RE.sub(" minus ", text)
    text = _MATH_PLUS_RE.sub(" plus ", text)
    text = _MATH_EQUALS_RE.sub(" equals ", text)
    text = _MATH_PERCENT_RE.sub(" percent", text)
    return text


# ---------------------------------------------------------------------------
# remaining numbers: thousands-separated and bare 4+ digit integers
# ---------------------------------------------------------------------------

# A tight numeric range with no surrounding space, e.g. "10-15" or
# "1,234-5,678". Runs before the single-number pass so both sides convert
# together instead of each being matched separately.
_RANGE_RE = re.compile(r"\b(\d{1,3}(?:,\d{3})*)-(\d{1,3}(?:,\d{3})*)\b")

# Thousands-grouped number ("12,345"), or a bare integer/decimal of 4+
# digits not already claimed by years/decades/clock/ordinals above. Numbers
# under 1000 are left as digits (a TTS engine reads short digit runs fine,
# and spelling them out is what turns "GPT-4" into a mess).
_NUMBER_RE = re.compile(r"\b\d{1,3}(?:,\d{3})+(?:\.\d+)?\b|\b\d{4,}(?:\.\d+)?\b")


def _cardinal(digits: str) -> str:
    cleaned = digits.replace(",", "")
    value: int | float = float(cleaned) if "." in cleaned else int(cleaned)
    return str(num2words(value))


def _range_repl(match: re.Match[str]) -> str:
    try:
        return f"{_cardinal(match.group(1))} to {_cardinal(match.group(2))}"
    except OverflowError:
        return match.group(0)


def _is_glued_to_a_word(text: str, start: int, end: int) -> bool:
    # A number stuck to a letter (directly, or across a hyphen) is an
    # identifier, not a quantity: "GPT-4", "COVID-19", "v2" must stay as-is.
    before = text[:start]
    after = text[end:]
    if before and before[-1].isalpha():
        return True
    if before and before[-1] == "-" and len(before) >= 2 and before[-2].isalpha():
        return True
    return bool(after and after[0].isalpha())


def numbers_to_words(text: str) -> str:
    text = _RANGE_RE.sub(_range_repl, text)

    def repl(match: re.Match[str]) -> str:
        if _is_glued_to_a_word(text, match.start(), match.end()):
            return match.group(0)
        try:
            return _cardinal(match.group(0))
        except OverflowError:
            # Also covers a decimal so long that float() rounds it to inf.
            return match.group(0)

    return _NUMBER_RE.sub(repl, text)


# ---------------------------------------------------------------------------
# abbreviations, punctuation, hyphenation
# ---------------------------------------------------------------------------

_MULTI_DASH_RE = re.compile(r"-{2,}")  # runs of hyphens (dashes already normalized to "-") -> one
_MULTI_SPACE_RE = re.compile(r"[ \t]{2,}")  # repeated spaces left behind by other substitutions

# A hyphen immediately followed by whitespace then a lowercase continuation
# is a line-wrap artifact ("self- pity"), not a real word break.
_HYPHEN_BREAK_RE = re.compile(r"(?<=\w)-\s+(?=[a-z])")


def expand_abbreviations(text: str) -> str:
    return ABBREV_RE.sub(lambda m: ABBREVIATIONS[m.group(1)], text)


def fix_punctuation(text: str) -> str:
    for old, new in PUNCTUATION_PAIRS:
        text = text.replace(old, new)
    text = _MULTI_DASH_RE.sub("-", text)
    text = _MULTI_SPACE_RE.sub(" ", text)
    return text.strip()


def fix_hyphenation(text: str) -> str:
    return _HYPHEN_BREAK_RE.sub("-", text)


# ---------------------------------------------------------------------------
# composition
# ---------------------------------------------------------------------------


def normalize_english(text: str) -> str:
    """Run the full English normalization pipeline.

    Order matters - each stage must not clobber a pattern a later stage
    still needs:

    1. fix_hyphenation first, so a word broken across a line wrap
       ("self- pity") is whole again before anything tries to read its
       pieces as separate tokens.
    2. decades_to_words before years_to_words: a decade like "1960s" has no
       word boundary right after its 4 digits (followed by "s"), so the
       year regex can't match it anyway, but converting decades first means
       there are no bare 4-digit spans left for the year heuristic to
       second-guess.
    3. years_to_words before roman_numerals_to_words/clock_to_words/
       numbers_to_words: once a year is spelled out ("nineteen ninety-
       seven") there are no digits left in that span for a later stage to
       misread as a plain cardinal. Doing this the other way round is the
       exact bug this order avoids - see the deliberately order-sensitive
       case in tests/text/test_normalize_english.py ("in 1997 he moved").
    4. roman_numerals_to_words before clock_to_words: independent in
       practice (no shared characters), kept in this slot since heading-like
       markers are resolved before anything numeric.
    5. clock_to_words before ordinals_and_math_to_words/numbers_to_words: a
       clock's ":" and digits must be consumed before the plain-number pass
       could otherwise treat "18" and "00" as two unrelated bare numbers.
    6. ordinals_and_math_to_words before numbers_to_words: "1st" must become
       "first" as a whole token before numbers_to_words could otherwise
       leave the glued "1" + "st" in a broken half-converted state.
    7. numbers_to_words last among the numeric stages: by now every digit
       span that needed special handling (year/decade/roman/clock/ordinal)
       has already been claimed, so whatever digits remain are safe to read
       as plain cardinals/ranges.
    8. expand_abbreviations and fix_punctuation last: abbreviation periods
       and curly quotes are cosmetic and don't interact with any numeric
       stage above.
    """
    text = fix_hyphenation(text)
    text = decades_to_words(text)
    text = years_to_words(text)
    text = roman_numerals_to_words(text)
    text = clock_to_words(text)
    text = ordinals_and_math_to_words(text)
    text = numbers_to_words(text)
    text = expand_abbreviations(text)
    text = fix_punctuation(text)
    return text
