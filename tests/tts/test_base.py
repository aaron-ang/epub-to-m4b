from __future__ import annotations

import numpy as np

from epub_to_m4b.tts.base import float32_to_pcm16, pcm16_to_float32


def test_pcm16_to_float32_round_trip_extremes() -> None:
    pcm = np.array([-32768, 0, 32767], dtype=np.int16)
    floats = pcm16_to_float32(pcm)
    assert floats.dtype == np.float32
    assert floats[1] == 0.0
    assert -1.0 <= floats[0] <= 1.0
    assert -1.0 <= floats[2] <= 1.0


def test_float32_to_pcm16_round_trip() -> None:
    floats = np.array([-1.0, 0.0, 0.5, 1.0], dtype=np.float32)
    pcm = float32_to_pcm16(floats)
    assert pcm.dtype == np.int16
    assert pcm[1] == 0
    assert pcm[2] > 0
    assert pcm[3] == 32767


def test_float32_to_pcm16_clips_out_of_range() -> None:
    floats = np.array([-2.0, 2.0], dtype=np.float32)
    pcm = float32_to_pcm16(floats)
    assert pcm[0] == -32768
    assert pcm[1] == 32767
