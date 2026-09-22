"""OpenAIEngine: the ``/v1/audio/speech`` endpoint shape shared by OpenAI
and the many servers that imitate it.

``base_url`` selects the server; everything else is the standard request:
one text in, one raw s16le PCM body out, so ``synthesize`` issues one POST
per sentence and decodes each body straight into a clip.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import ClassVar

import httpx

from epub_to_m4b.book import AudioClip
from epub_to_m4b.tts.base import TTSEngine, fingerprint_digest
from epub_to_m4b.tts.http import new_client, pcm_response_to_clip, post_with_retry

# Rate of the raw PCM the API returns for ``response_format=pcm``.
_SAMPLE_RATE = 24000


@dataclass(frozen=True)
class OpenAIConfig:
    base_url: str
    model: str
    voice: str
    speed: float = 1.0
    api_key_env: str = "OPENAI_API_KEY"


@dataclass
class OpenAIEngine(TTSEngine):
    name: ClassVar[str] = "openai"

    config: OpenAIConfig
    api_key: str
    transport: httpx.BaseTransport | None = None
    sleep: Callable[[float], None] = time.sleep

    sample_rate: int = field(default=_SAMPLE_RATE, init=False)
    max_concurrency: int = field(default=4, init=False)
    _client: httpx.Client = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._client = new_client(self.transport)

    def synthesize(self, texts: Sequence[str]) -> list[AudioClip]:
        url = f"{self.config.base_url.rstrip('/')}/v1/audio/speech"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        clips = []
        for text in texts:
            body = {
                "model": self.config.model,
                "input": text,
                "voice": self.config.voice,
                "speed": self.config.speed,
                "response_format": "pcm",
            }
            response = post_with_retry(
                self._client, url, sleep=self.sleep, headers=headers, json=body
            )
            clips.append(pcm_response_to_clip(response, self.sample_rate))
        return clips

    def fingerprint(self) -> str:
        return fingerprint_digest(
            self.name, self.config.model, self.config.voice, repr(self.config.speed)
        )

    def close(self) -> None:
        self._client.close()
