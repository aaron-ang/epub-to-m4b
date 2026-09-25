"""Shared TTS engine interface, PCM16 decoding, fingerprint digest.

Real engines exchange audio with their HTTP API or local sidecar as 16-bit
PCM; everything else in this project works in ``AudioClip``'s float32
domain. ``pcm16_to_float32`` is the shared boundary so no engine module
reimplements the scaling. ``fingerprint_digest`` is likewise the one place
that turns an engine's audio-affecting settings into the cache-partitioning
string.
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from collections.abc import Sequence
from types import TracebackType
from typing import ClassVar, Self

import numpy as np
import numpy.typing as npt

from epub_to_m4b.book import AudioClip

_PCM16 = np.iinfo(np.int16)
# Full-scale magnitude: float 1.0 maps to the int16 range's negative bound.
_PCM16_SCALE = float(-_PCM16.min)


def pcm16_to_float32(samples: npt.NDArray[np.int16]) -> npt.NDArray[np.float32]:
    return (samples.astype(np.float32) / _PCM16_SCALE).astype(np.float32)


def fingerprint_digest(*parts: str) -> str:
    """sha256 over NUL-joined ``parts`` - the separator keeps adjacent parts
    from running together into the same digest."""
    digest = hashlib.sha256()
    for part in parts:
        digest.update(part.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


class TTSEngine(ABC):
    name: ClassVar[str]
    sample_rate: int
    max_batch: int = 1
    max_concurrency: int = 1
    # Reseeds allowed per clip that ``clip_miss`` rejects; 0 disables retries.
    retries: int = 0

    @abstractmethod
    def synthesize(self, texts: Sequence[str]) -> list[AudioClip]: ...

    def clip_miss(self, text: str, clip: AudioClip, *, capped: bool) -> float:
        """How far ``clip`` falls from plausible speech for ``text``; 0.0 keeps it.

        ``capped`` marks a clip from a request with a generation cap
        (``synthesize``, or ``resynthesize`` with ``capped=True``); a clip the
        cap may have stopped mid-word should score ``math.inf``. Only
        consulted when ``retries`` is above zero.
        """
        return 0.0

    def resynthesize(
        self, texts: Sequence[str], retry_round: int, *, capped: bool
    ) -> list[AudioClip]:
        """Fresh takes of ``texts`` for retry round ``retry_round`` (1, 2, ...).

        Each round must draw differently from the first pass and from every
        other round. ``capped=False`` lifts the generation cap so no take is
        stopped early. Only called when ``retries`` is above zero.
        """
        raise NotImplementedError(f"{self.name} does not retry clips")

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
