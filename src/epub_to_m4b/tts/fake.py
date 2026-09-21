"""Fake TTS engines with no network/GPU dependency: silence and an audible tone.

Used for tests and for ``--engine silence`` dry runs that exercise the full
pipeline (assembly, chapter markers, VTT) without any real speech model.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from epub_to_m4b.book import AudioClip
from epub_to_m4b.tts.base import TTSEngine

# Not meant to approximate real speech pacing - just deterministic and long
# enough that downstream VTT cues have a meaningful, non-zero duration.
_CHARS_PER_SECOND = 15.0
_MIN_SECONDS = 0.2


def _duration_seconds(text: str) -> float:
    return max(_MIN_SECONDS, len(text) / _CHARS_PER_SECOND)


@dataclass
class SilenceEngine(TTSEngine):
    name: ClassVar[str] = "silence"
    sample_rate: int = 24000

    def synthesize(self, texts: Sequence[str]) -> list[AudioClip]:
        clips = []
        for text in texts:
            n_samples = int(_duration_seconds(text) * self.sample_rate)
            samples = np.zeros(n_samples, dtype=np.float32)
            clips.append(AudioClip(samples=samples, sample_rate=self.sample_rate))
        return clips

    def fingerprint(self) -> str:
        return f"silence:{self.sample_rate}"


@dataclass
class ToneEngine(TTSEngine):
    name: ClassVar[str] = "tone"
    sample_rate: int = 24000
    frequency: float = 440.0
    amplitude: float = 0.2

    def synthesize(self, texts: Sequence[str]) -> list[AudioClip]:
        clips = []
        for text in texts:
            n_samples = int(_duration_seconds(text) * self.sample_rate)
            t = np.arange(n_samples, dtype=np.float32) / self.sample_rate
            samples = (self.amplitude * np.sin(2 * np.pi * self.frequency * t)).astype(np.float32)
            clips.append(AudioClip(samples=samples, sample_rate=self.sample_rate))
        return clips

    def fingerprint(self) -> str:
        return f"tone:{self.sample_rate}:{self.frequency}"
