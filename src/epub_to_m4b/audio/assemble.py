"""Sentence/clip pairs -> one chapter audio file, with silence gaps between.

M3 keeps this to a single in-memory concatenation per chapter (books are
small enough that this is not a problem); a real streaming writer that
appends to a ``soundfile.SoundFile`` block-by-block would be a drop-in
replacement here if a chapter ever got too large to hold in memory.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import numpy.typing as npt
import soundfile as sf

from epub_to_m4b.book import AudioClip, Sentence

_MIN_CHAPTER_SECONDS = 0.1


def silence(seconds: float, sample_rate: int) -> npt.NDArray[np.float32]:
    return np.zeros(int(seconds * sample_rate), dtype=np.float32)


def assemble_chapter(
    pairs: Sequence[tuple[Sentence, AudioClip]],
    out_path: Path,
    *,
    sample_rate: int,
) -> list[tuple[float, float]]:
    """Write the chapter to ``out_path`` and return each sentence's (start, end) offset."""
    offsets: list[tuple[float, float]] = []
    chunks: list[npt.NDArray[np.float32]] = []
    cursor = 0.0
    for sentence, clip in pairs:
        if clip.sample_rate != sample_rate:
            raise ValueError(f"clip sample_rate {clip.sample_rate} != chapter rate {sample_rate}")
        start = cursor
        chunks.append(clip.samples)
        cursor += clip.seconds
        offsets.append((start, cursor))
        if sentence.gap_after > 0:
            chunks.append(silence(sentence.gap_after, sample_rate))
            cursor += sentence.gap_after

    audio = np.concatenate(chunks) if chunks else silence(_MIN_CHAPTER_SECONDS, sample_rate)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(out_path, audio, sample_rate)
    return offsets
