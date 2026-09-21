"""Split a paragraph's text into sentence-sized chunks for the TTS engine.

Splits on sentence punctuation, force-cuts anything still too long at the
best available punctuation/space, then merges orphan-short fragments back
into a neighbor.

This module only produces raw sentence text. A later milestone wraps each
string into a ``Sentence`` with a computed ``gap_after`` - the orchestrator
owns gaps between clips, not the splitter, so that logic is out of scope
here.
"""

from __future__ import annotations

import re

from epub_to_m4b.book import Paragraph
from epub_to_m4b.text.lang.tables_en import ABBREVIATIONS

# Sentence-ending punctuation. "." gets extra scrutiny below (abbreviations,
# decimals); ! ? ; : always end a sentence wherever they appear.
_BOUNDARY_CHARS = ".!?;:"

# A closing quote/bracket that belongs with the sentence that just ended,
# e.g. the `"` in `He said "stop."` - the boundary is after it, not before.
_CLOSERS = "\"')]\u201d\u2019"

_ABBREV_TOKENS = tuple(sorted(ABBREVIATIONS, key=len, reverse=True))
# Matches a known abbreviation ("Mr.", "e.g.", ...) as a whole token, so
# every period inside it (including the internal ones in "e.g.") can be
# marked as "not a sentence boundary" below.
_ABBREV_RE = re.compile(r"(?<!\w)(" + "|".join(re.escape(tok) for tok in _ABBREV_TOKENS) + r")")


def _protected_periods(text: str) -> set[int]:
    """Indices of '.' characters that belong to a known abbreviation."""
    protected: set[int] = set()
    for match in _ABBREV_RE.finditer(text):
        for offset, ch in enumerate(match.group(0)):
            if ch == ".":
                protected.add(match.start() + offset)
    return protected


def _sentence_end_positions(text: str) -> list[int]:
    protected = _protected_periods(text)
    n = len(text)
    positions: list[int] = []
    i = 0
    while i < n:
        ch = text[i]
        if ch in _BOUNDARY_CHARS:
            if ch == "." and i in protected:
                i += 1
                continue
            prev_digit = i > 0 and text[i - 1].isdigit()
            next_digit = i + 1 < n and text[i + 1].isdigit()
            if ch == "." and prev_digit and next_digit:
                # Decimal point ("3.14"): digit immediately on both sides.
                i += 1
                continue
            end = i + 1
            while end < n and text[end] in _CLOSERS:
                end += 1
            if end >= n or text[end].isspace():
                positions.append(end)
                i = end
                continue
        i += 1
    return positions


def _raw_sentences(text: str) -> list[str]:
    positions = _sentence_end_positions(text)
    pieces: list[str] = []
    last = 0
    for end in positions:
        piece = text[last:end].strip()
        if piece:
            pieces.append(piece)
        last = end
    tail = text[last:].strip()
    if tail:
        pieces.append(tail)
    return pieces


def _cut_long(piece: str, max_chars: int) -> list[str]:
    if len(piece) <= max_chars:
        return [piece]
    window = piece[: max_chars + 1]
    # Preference 1: the last comma/semicolon before the limit, kept with the
    # left half so the pause falls where the punctuation already implies one.
    comma_pos = window.rfind(",")
    semi_pos = window.rfind(";")
    cut = max(comma_pos, semi_pos)
    if cut > 0:
        idx = cut + 1
    else:
        # Preference 2: the last space before the limit.
        idx = window.rfind(" ")
        if idx <= 0:
            # Preference 3 (last resort): a hard cut mid-word at the limit.
            idx = max_chars
    left = piece[:idx].strip()
    right = piece[idx:].strip()
    if not left or not right:
        return [piece.strip()]
    return [left, *_cut_long(right, max_chars)]


def _merge_short(pieces: list[str], max_chars: int) -> list[str]:
    threshold = max_chars / 2
    ceiling = max_chars * 1.5
    merged: list[str] = []
    i = 0
    n = len(pieces)
    while i < n:
        current = pieces[i]
        if (
            len(current) < threshold
            and i + 1 < n
            and len(current) + 1 + len(pieces[i + 1]) <= ceiling
        ):
            merged.append(f"{current} {pieces[i + 1]}")
            i += 2
            continue
        merged.append(current)
        i += 1
    return merged


def split_paragraph(paragraph: Paragraph, *, max_chars: int = 125) -> list[str]:
    text = paragraph.text.strip()
    if not text:
        return []
    raw = _raw_sentences(text)
    cut: list[str] = []
    for piece in raw:
        cut.extend(_cut_long(piece, max_chars))
    return _merge_short(cut, max_chars)
