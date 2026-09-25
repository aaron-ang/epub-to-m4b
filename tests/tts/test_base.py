from __future__ import annotations

import numpy as np

from epub_to_m4b.tts.base import pcm16_to_float32


def test_pcm16_to_float32_round_trip_extremes() -> None:
    pcm = np.array([-32768, 0, 32767], dtype=np.int16)
    floats = pcm16_to_float32(pcm)
    assert floats.dtype == np.float32
    assert floats[1] == 0.0
    assert -1.0 <= floats[0] <= 1.0
    assert -1.0 <= floats[2] <= 1.0
