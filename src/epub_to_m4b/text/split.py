"""Split a paragraph's normalized text into clip-sized pieces for the TTS engine.

Cuts at sentence ends, force-cuts any sentence still over the limit at its
best comma, semicolon or space, drops pieces with nothing to say, then fills
each clip with consecutive pieces up to the limit.

This module only produces raw text. The orchestrator wraps each string into a
``Sentence`` with a computed ``gap_after``; gaps between clips are its job.
"""

from __future__ import annotations

import re
from itertools import pairwise

from epub_to_m4b.book import Paragraph

# A closing quote/bracket that belongs with the sentence that just ended,
# e.g. the `"` in `He said "stop."` - the boundary is after it, not before.
# Public: the orchestrator's gap rule looks past the same characters to find
# a sentence's real terminator.
CLOSERS = "\"')]\u201d\u2019"

# Longest clip the splitter hands to an engine. Long enough to keep a full
# clause's prosody in one clip; short enough that the Breeze runaway guard
# (tts/guard.py scales its limits per character) stays tight and every
# provider's per-request cap is far away.
DEFAULT_MAX_CHARS = 125

# ! ? ; : or a full stop, plus any closers, followed by whitespace or the
# end. A full stop after a lone capital ("J. K. Rowling", "S. Place") is a
# name initial; a one-letter sentence-final word ("Plan B. Then") is misread
# the same way, and the length cut still bounds it.
_SENTENCE_END = re.compile(rf"(?:[!?;:]|(?<!\b[A-Z])\.)[{re.escape(CLOSERS)}]*(?=\s|$)")


def _sentences(text: str) -> list[str]:
    ends = [m.end() for m in _SENTENCE_END.finditer(text)]
    return [s for a, b in pairwise([0, *ends, len(text)]) if (s := text[a:b].strip())]


def _cut_long(piece: str, max_chars: int) -> list[str]:
    """Cut after the last comma or semicolon within the limit, so the pause
    falls where the punctuation already implies one; else at the last space;
    else mid-word at the limit."""
    pieces = []
    while len(piece) > max_chars:
        # The punctuation stays left of the cut, a space is dropped at it.
        head = piece[:max_chars]
        cut = max(head.rfind(","), head.rfind(";")) + 1 or piece.rfind(" ", 0, max_chars + 1)
        if cut <= 0:
            cut = max_chars
        pieces.append(piece[:cut].strip())
        piece = piece[cut:].strip()
    return [*pieces, piece] if piece else pieces


def _speakable(piece: str) -> bool:
    """Whether a piece has a letter or digit. One without (a stray closing
    quote, a "* * *" separator) gives an engine nothing to say."""
    return any(ch.isalnum() for ch in piece)


def _fill(pieces: list[str], max_chars: int) -> list[str]:
    clips: list[str] = []
    for piece in pieces:
        if clips and len(clips[-1]) + 1 + len(piece) <= max_chars:
            clips[-1] = f"{clips[-1]} {piece}"
        else:
            clips.append(piece)
    return clips


def split_paragraph(paragraph: Paragraph, *, max_chars: int = DEFAULT_MAX_CHARS) -> list[str]:
    pieces = [c for s in _sentences(paragraph.text.strip()) for c in _cut_long(s, max_chars)]
    return _fill([p for p in pieces if _speakable(p)], max_chars)
