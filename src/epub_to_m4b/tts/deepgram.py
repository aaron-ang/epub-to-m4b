"""DeepgramEngine: Deepgram Aura text-to-speech over ``/v1/speak``.

Model and output format are query parameters, so the JSON body is just the
text. ``encoding=linear16`` with ``container=none`` yields raw s16le at the
requested sample rate.
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

# Rate requested via the ``sample_rate=24000`` query parameter.
_SAMPLE_RATE = 24000


@dataclass(frozen=True)
class DeepgramConfig:
    model: str = "aura-2-thalia-en"
    base_url: str = "https://api.deepgram.com"
    api_key_env: str = "DEEPGRAM_API_KEY"


@dataclass
class DeepgramEngine(TTSEngine):
    name: ClassVar[str] = "deepgram"

    config: DeepgramConfig
    api_key: str
    transport: httpx.BaseTransport | None = None
    sleep: Callable[[float], None] = time.sleep

    sample_rate: int = field(default=_SAMPLE_RATE, init=False)
    max_concurrency: int = field(default=4, init=False)
    _client: httpx.Client = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._client = new_client(self.transport)

    def synthesize(self, texts: Sequence[str]) -> list[AudioClip]:
        url = f"{self.config.base_url.rstrip('/')}/v1/speak"
        headers = {"Authorization": f"Token {self.api_key}"}
        params = {
            "model": self.config.model,
            "encoding": "linear16",
            "sample_rate": str(self.sample_rate),
            "container": "none",
        }
        clips = []
        for text in texts:
            response = post_with_retry(
                self._client,
                url,
                sleep=self.sleep,
                headers=headers,
                params=params,
                json={"text": text},
            )
            clips.append(pcm_response_to_clip(response, self.sample_rate))
        return clips

    def fingerprint(self) -> str:
        return fingerprint_digest(self.name, self.config.model)

    def close(self) -> None:
        self._client.close()
