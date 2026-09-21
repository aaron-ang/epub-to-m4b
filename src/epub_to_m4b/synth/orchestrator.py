"""Sentences -> batches -> engine, with gap policy in between.

No cache lookups here yet - that's ``synth/cache.py`` in a later milestone.
For now every sentence is always synthesized: normalize + split each
paragraph into sentence-sized strings, attach a deterministic
``gap_after`` per the punctuation/position rules below, then batch and
call the engine.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from epub_to_m4b.book import AudioClip, Chapter, Paragraph, ParagraphKind, Sentence
from epub_to_m4b.text.normalize import normalize
from epub_to_m4b.text.split import split_paragraph
from epub_to_m4b.tts.base import TTSEngine

# A closing quote/bracket trailing the real terminator, e.g. the `"` in
# `He said "stop."` - look past it to find what actually ended the sentence.
_CLOSERS = "\"')]\u201d\u2019"
_CLAUSE_CHARS = ",;:"


@dataclass(frozen=True, slots=True)
class GapPolicy:
    sentence_end: float = 0.25
    clause: float = 0.10
    paragraph: float = 0.40
    heading: float = 0.80


_DEFAULT_POLICY = GapPolicy()


def sentence_gap(
    text: str,
    kind: ParagraphKind,
    *,
    is_last_in_paragraph: bool,
    is_last_in_chapter: bool,
    policy: GapPolicy = _DEFAULT_POLICY,
) -> float:
    """The gap that follows one sentence, per the punctuation/position rules.

    Heading sentences always get the heading gap. A paragraph-closing
    sentence gets the paragraph gap, unless it also closes the chapter -
    the chapter boundary itself provides the separation, so that case falls
    through to the ordinary punctuation-based gap instead of stacking an
    extra pause.
    """
    if kind is ParagraphKind.HEADING:
        return policy.heading
    if is_last_in_paragraph and not is_last_in_chapter:
        return policy.paragraph
    return policy.clause if _terminal_char(text) in _CLAUSE_CHARS else policy.sentence_end


def _terminal_char(text: str) -> str:
    stripped = text.rstrip()
    while stripped and stripped[-1] in _CLOSERS:
        stripped = stripped[:-1]
    return stripped[-1] if stripped else ""


def chapter_to_sentences(
    chapter: Chapter,
    chapter_index: int,
    *,
    policy: GapPolicy = _DEFAULT_POLICY,
    lang: str = "en",
) -> list[Sentence]:
    flat: list[tuple[str, ParagraphKind, bool]] = []
    for paragraph in chapter.paragraphs:
        normalized = normalize(paragraph.text, lang=lang)
        pieces = split_paragraph(Paragraph(text=normalized, kind=paragraph.kind))
        last = len(pieces) - 1
        for i, piece in enumerate(pieces):
            flat.append((piece, paragraph.kind, i == last))

    last_index = len(flat) - 1
    sentences: list[Sentence] = []
    for i, (text, kind, is_last_in_paragraph) in enumerate(flat):
        gap = sentence_gap(
            text,
            kind,
            is_last_in_paragraph=is_last_in_paragraph,
            is_last_in_chapter=i == last_index,
            policy=policy,
        )
        sentences.append(Sentence(text=text, gap_after=gap, chapter_index=chapter_index))
    return sentences


def batch_sentences(sentences: Sequence[Sentence], max_batch: int) -> list[list[Sentence]]:
    return [list(sentences[i : i + max_batch]) for i in range(0, len(sentences), max_batch)]


def synthesize_chapter(
    sentences: Sequence[Sentence], engine: TTSEngine
) -> list[tuple[Sentence, AudioClip]]:
    pairs: list[tuple[Sentence, AudioClip]] = []
    for batch in batch_sentences(sentences, engine.max_batch):
        clips = engine.synthesize([s.text for s in batch])
        if len(clips) != len(batch):
            raise ValueError(f"engine returned {len(clips)} clips for {len(batch)} texts")
        pairs.extend(zip(batch, clips, strict=True))
    return pairs
