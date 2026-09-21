from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

from epub_to_m4b.audio.assemble import assemble_chapter, silence
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
    offsets = assemble_chapter(pairs, out_path, sample_rate=_RATE)
    assert offsets == [(0.0, 1.0), (1.5, 3.5)]
    assert out_path.is_file()
    data, rate = sf.read(out_path)
    assert rate == _RATE
    assert len(data) == int(3.5 * _RATE)


def test_assemble_chapter_empty_pairs_still_writes_file(tmp_path: Path) -> None:
    out_path = tmp_path / "empty.flac"
    offsets = assemble_chapter([], out_path, sample_rate=_RATE)
    assert offsets == []
    assert out_path.is_file()
    data, rate = sf.read(out_path)
    assert rate == _RATE
    assert len(data) > 0
