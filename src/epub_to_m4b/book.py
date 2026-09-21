"""Core data model shared by every stage of the pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np
import numpy.typing as npt


class ParagraphKind(StrEnum):
    HEADING = "heading"
    BODY = "body"
    TABLE_ROW = "row"


@dataclass(frozen=True, slots=True)
class Paragraph:
    text: str
    kind: ParagraphKind


@dataclass(frozen=True, slots=True)
class Chapter:
    title: str
    paragraphs: tuple[Paragraph, ...]
    source_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Book:
    title: str
    author: str | None
    cover: bytes | None
    cover_mime: str | None
    chapters: tuple[Chapter, ...]
    source_sha256: str


@dataclass(frozen=True, slots=True)
class Sentence:
    text: str
    gap_after: float
    chapter_index: int


@dataclass(frozen=True, slots=True)
class AudioClip:
    samples: npt.NDArray[np.float32]  # mono, shape (n,)
    sample_rate: int

    @property
    def seconds(self) -> float:
        return len(self.samples) / self.sample_rate
