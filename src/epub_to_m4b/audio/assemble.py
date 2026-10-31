"""Sentence/clip pairs -> one chapter audio file, with silence gaps between.

Each clip's own leading and trailing silence is trimmed first, so the gaps
are the only pauses between sentences whatever the engine left at its edges.

Each chapter is concatenated in memory in one pass (chapters are small
enough for this); a streaming writer that appends to a
``soundfile.SoundFile`` block-by-block would be a drop-in replacement here
if a chapter ever got too large to hold in memory.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import numpy.typing as npt
import soundfile as sf

from epub_to_m4b.book import AudioClip, Sentence

_MIN_CHAPTER_SECONDS = 0.1
# Engines leave their own silence at clip edges, from none to seconds
# depending on engine and voice. An _EDGE_FRAME_SECONDS frame quieter than the clip's
# loudest frame by more than _EDGE_SILENCE_DB is silence; trimming stops
# _EDGE_PAD_SECONDS short of the outermost loud frames so breaths and
# decaying word endings survive.
_EDGE_FRAME_SECONDS = 0.01
_EDGE_SILENCE_DB = -40.0
_EDGE_PAD_SECONDS = 0.05


def silence(seconds: float, sample_rate: int) -> npt.NDArray[np.float32]:
    return np.zeros(int(seconds * sample_rate), dtype=np.float32)


def trim_edges(samples: npt.NDArray[np.float32], sample_rate: int) -> npt.NDArray[np.float32]:
    """``samples`` without leading and trailing silence beyond the pad; an
    all-silent or non-finite clip comes back unchanged."""
    frame = max(1, round(_EDGE_FRAME_SECONDS * sample_rate))
    frames = len(samples) // frame
    if frames == 0:
        return samples

    # Zero-pad the last partial frame so speech in it still counts.
    padded = np.zeros(-(-len(samples) // frame) * frame, dtype=np.float32)
    padded[: len(samples)] = samples
    rms = np.sqrt(np.mean(np.square(padded.reshape(-1, frame)), axis=1))
    peak = float(rms.max())
    if peak == 0.0 or not np.isfinite(peak):
        return samples

    loud = np.flatnonzero(rms >= peak * 10 ** (_EDGE_SILENCE_DB / 20))
    pad = round(_EDGE_PAD_SECONDS * sample_rate)
    start = max(0, int(loud[0]) * frame - pad)
    end = min(len(samples), (int(loud[-1]) + 1) * frame + pad)
    return samples[start:end]


def assemble_chapter(
    pairs: Sequence[tuple[Sentence, AudioClip]],
    out_path: Path,
    *,
    sample_rate: int,
) -> tuple[list[tuple[float, float]], float]:
    """Write the chapter to ``out_path``, each clip trimmed by ``trim_edges``.

    Returns each sentence's (start, end) offset and the chapter's total
    duration. The total includes the last sentence's trailing gap (if any),
    since that silence is written into the file too - callers must use this
    total, not the last sentence's own ``end`` offset, when placing the next
    chapter's start: undercounting the trailing gap here would leave every
    later chapter's marker pointing earlier than where its audio actually
    starts in the concatenated file.
    """
    offsets: list[tuple[float, float]] = []
    chunks: list[npt.NDArray[np.float32]] = []
    cursor = 0.0
    for sentence, clip in pairs:
        if clip.sample_rate != sample_rate:
            raise ValueError(f"clip sample_rate {clip.sample_rate} != chapter rate {sample_rate}")
        start = cursor
        samples = trim_edges(clip.samples, sample_rate)
        chunks.append(samples)
        cursor += len(samples) / sample_rate
        offsets.append((start, cursor))
        if sentence.gap_after > 0:
            chunks.append(silence(sentence.gap_after, sample_rate))
            cursor += sentence.gap_after

    audio = np.concatenate(chunks) if chunks else silence(_MIN_CHAPTER_SECONDS, sample_rate)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(out_path, audio, sample_rate)
    duration = len(audio) / sample_rate
    return offsets, duration
