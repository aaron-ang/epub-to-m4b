from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

from epub_to_m4b.audio.assemble import assemble_chapter, silence, trim_edges
from epub_to_m4b.book import AudioClip, Sentence

_RATE = 8000


def _clip(seconds: float) -> AudioClip:
    return AudioClip(samples=np.ones(int(seconds * _RATE), dtype=np.float32), sample_rate=_RATE)


def test_silence_helper_length_and_dtype() -> None:
    samples = silence(0.5, _RATE)
    assert len(samples) == int(0.5 * _RATE)
    assert samples.dtype == np.float32
    assert (samples == 0).all()


def test_assemble_chapter_computes_offsets_with_gaps(tmp_path: Path) -> None:
    pairs = [
        (Sentence(text="a", gap_after=0.5, chapter_index=0), _clip(1.0)),
        (Sentence(text="b", gap_after=0.0, chapter_index=0), _clip(2.0)),
    ]
    out_path = tmp_path / "chapter.flac"
    offsets, duration = assemble_chapter(pairs, out_path, sample_rate=_RATE)
    assert offsets == [(0.0, 1.0), (1.5, 3.5)]
    assert duration == 3.5
    assert out_path.is_file()
    data, rate = sf.read(out_path)
    assert rate == _RATE
    assert len(data) == int(3.5 * _RATE)


def test_assemble_chapter_duration_includes_trailing_gap_after_last_sentence(
    tmp_path: Path,
) -> None:
    # The last sentence's gap_after is still written into the file (a pause
    # before the next chapter's audio starts) - the returned duration must
    # include it too, or every later chapter's marker drifts out of sync with
    # where its audio actually begins in the concatenated book.
    pairs = [(Sentence(text="a", gap_after=0.25, chapter_index=0), _clip(1.0))]
    out_path = tmp_path / "chapter.flac"
    offsets, duration = assemble_chapter(pairs, out_path, sample_rate=_RATE)
    assert offsets == [(0.0, 1.0)]
    assert duration == 1.25
    data, _rate = sf.read(out_path)
    assert len(data) == int(1.25 * _RATE)


def test_assemble_chapter_empty_pairs_still_writes_file(tmp_path: Path) -> None:
    out_path = tmp_path / "empty.flac"
    offsets, duration = assemble_chapter([], out_path, sample_rate=_RATE)
    assert offsets == []
    assert duration > 0
    assert out_path.is_file()
    data, rate = sf.read(out_path)
    assert rate == _RATE
    assert len(data) > 0


def _padded(lead: float, speech: float, trail: float, *, floor: float = 0.0) -> AudioClip:
    """``speech`` seconds at full scale between ``lead`` and ``trail`` seconds
    at ``floor``."""
    parts = [
        np.full(int(lead * _RATE), floor, dtype=np.float32),
        np.ones(int(speech * _RATE), dtype=np.float32),
        np.full(int(trail * _RATE), floor, dtype=np.float32),
    ]
    return AudioClip(samples=np.concatenate(parts), sample_rate=_RATE)


def test_trim_edges_keeps_a_pad_around_the_speech() -> None:
    trimmed = trim_edges(_padded(0.5, 1.0, 2.0).samples, _RATE)
    assert len(trimmed) == int((0.05 + 1.0 + 0.05) * _RATE)


def test_trim_edges_treats_a_quiet_floor_as_silence() -> None:
    # 1e-3 is 60 dB under the speech: room noise, not speech.
    trimmed = trim_edges(_padded(0.5, 1.0, 0.5, floor=1e-3).samples, _RATE)
    assert len(trimmed) == int(1.1 * _RATE)


def test_trim_edges_keeps_quiet_speech_within_the_threshold() -> None:
    # 0.1 is 20 dB under the peak: a soft word ending, kept whole.
    clip = _padded(0.5, 1.0, 0.5, floor=0.1)
    assert len(trim_edges(clip.samples, _RATE)) == len(clip.samples)


def test_trim_edges_leaves_silent_and_tiny_clips_alone() -> None:
    silent = np.zeros(_RATE, dtype=np.float32)
    assert len(trim_edges(silent, _RATE)) == _RATE
    tiny = np.ones(10, dtype=np.float32)
    assert len(trim_edges(tiny, _RATE)) == 10


def test_assemble_chapter_offsets_follow_the_trimmed_clips(tmp_path: Path) -> None:
    pairs = [
        (Sentence(text="a", gap_after=0.5, chapter_index=0), _padded(0.0, 1.0, 1.95)),
        (Sentence(text="b", gap_after=0.0, chapter_index=0), _clip(1.0)),
    ]
    offsets, duration = assemble_chapter(pairs, tmp_path / "chapter.flac", sample_rate=_RATE)
    assert offsets == [(0.0, 1.05), (1.55, 2.55)]
    assert duration == 2.55


def test_trim_edges_counts_speech_in_the_last_partial_frame() -> None:
    # 80 samples is one 10 ms frame at 8 kHz; speech sits in a 40-sample tail.
    samples = np.zeros(2 * 80 + 40, dtype=np.float32)
    samples[-40:] = 1.0
    trimmed = trim_edges(samples, _RATE)
    assert trimmed[-1] == 1.0


def test_trim_edges_leaves_non_finite_clips_alone() -> None:
    samples = _padded(0.5, 1.0, 0.5).samples.copy()
    samples[10] = np.nan
    assert len(trim_edges(samples, _RATE)) == len(samples)
