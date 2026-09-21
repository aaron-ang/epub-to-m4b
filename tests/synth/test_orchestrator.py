from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np

from epub_to_m4b.book import AudioClip, Chapter, Paragraph, ParagraphKind, Sentence
from epub_to_m4b.synth.orchestrator import (
    GapPolicy,
    batch_sentences,
    chapter_to_sentences,
    sentence_gap,
    synthesize_chapter,
)
from epub_to_m4b.tts.base import TTSEngine

_S1 = "This is a considerably long first sentence that should not get merged."
_S2 = "This is a second, equally long sentence that also should not get merged."


def test_heading_sentence_gets_heading_gap() -> None:
    chapter = Chapter(
        title="Chapter One",
        paragraphs=(
            Paragraph(text="Chapter One", kind=ParagraphKind.HEADING),
            Paragraph(text=_S1, kind=ParagraphKind.BODY),
        ),
        source_ids=("c1",),
    )
    sentences = chapter_to_sentences(chapter, 0)
    assert sentences[0].text == "Chapter One"
    assert sentences[0].gap_after == GapPolicy().heading


def test_mid_paragraph_sentence_gets_sentence_end_gap() -> None:
    chapter = Chapter(
        title="Ch",
        paragraphs=(Paragraph(text=f"{_S1} {_S2}", kind=ParagraphKind.BODY),),
        source_ids=("c1",),
    )
    sentences = chapter_to_sentences(chapter, 0)
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
    sentences = chapter_to_sentences(chapter, 0)
    assert sentences[0].text == _S1
    assert sentences[0].gap_after == GapPolicy().paragraph


def test_last_sentence_of_chapter_falls_back_to_punctuation_gap() -> None:
    chapter = Chapter(
        title="Ch",
        paragraphs=(
            Paragraph(text=_S1, kind=ParagraphKind.BODY),
            Paragraph(text=_S2, kind=ParagraphKind.BODY),
        ),
        source_ids=("c1",),
    )
    sentences = chapter_to_sentences(chapter, 0)
    assert sentences[-1].text == _S2
    assert sentences[-1].gap_after == GapPolicy().sentence_end


def test_chapter_index_is_recorded_on_every_sentence() -> None:
    chapter = Chapter(
        title="Ch", paragraphs=(Paragraph(text=_S1, kind=ParagraphKind.BODY),), source_ids=("c1",)
    )
    sentences = chapter_to_sentences(chapter, 7)
    assert all(s.chapter_index == 7 for s in sentences)


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


def test_batch_sentences_respects_max_batch() -> None:
    sentences = [Sentence(text=f"s{i}", gap_after=0.0, chapter_index=0) for i in range(5)]
    batches = batch_sentences(sentences, 2)
    assert [len(b) for b in batches] == [2, 2, 1]
    assert [s.text for s in batches[0]] == ["s0", "s1"]
    assert [s.text for s in batches[2]] == ["s4"]


@dataclass
class _RecordingEngine(TTSEngine):
    name: ClassVar[str] = "recording"
    sample_rate: int = 8000
    max_batch: int = 2
    calls: list[list[str]] = field(default_factory=list)

    def synthesize(self, texts: Sequence[str]) -> list[AudioClip]:
        self.calls.append(list(texts))
        samples = np.zeros(1, dtype=np.float32)
        return [AudioClip(samples=samples, sample_rate=self.sample_rate) for _ in texts]

    def fingerprint(self) -> str:
        return "recording"


def test_synthesize_chapter_batches_and_pairs_in_order() -> None:
    sentences = [Sentence(text=f"s{i}", gap_after=0.0, chapter_index=0) for i in range(5)]
    engine = _RecordingEngine()
    pairs = synthesize_chapter(sentences, engine)
    assert [s.text for s, _clip in pairs] == ["s0", "s1", "s2", "s3", "s4"]
    assert engine.calls == [["s0", "s1"], ["s2", "s3"], ["s4"]]
