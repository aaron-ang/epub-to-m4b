"""Shared TTS engine interface and PCM16/float32 conversion helpers.

Real engines (later milestones) exchange audio with their HTTP/local sidecar
as 16-bit PCM; everything else in this project works in ``AudioClip``'s
float32 domain. These converters are the shared boundary so no engine module
reimplements the scaling.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from types import TracebackType
from typing import ClassVar, Self

import numpy as np
import numpy.typing as npt

from epub_to_m4b.book import AudioClip

_PCM16_SCALE = 32768.0


def pcm16_to_float32(samples: npt.NDArray[np.int16]) -> npt.NDArray[np.float32]:
    return (samples.astype(np.float32) / _PCM16_SCALE).astype(np.float32)


def float32_to_pcm16(samples: npt.NDArray[np.float32]) -> npt.NDArray[np.int16]:
    scaled = np.clip(samples, -1.0, 1.0) * _PCM16_SCALE
    return np.clip(np.round(scaled), -_PCM16_SCALE, _PCM16_SCALE - 1).astype(np.int16)


class TTSEngine(ABC):
    name: ClassVar[str]
    sample_rate: int
    max_batch: int = 1
    max_concurrency: int = 1
    max_chars: int = 4096

    @abstractmethod
    def synthesize(self, texts: Sequence[str]) -> list[AudioClip]: ...

    @abstractmethod
    def fingerprint(self) -> str: ...

    def close(self) -> None:
        return None

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()
