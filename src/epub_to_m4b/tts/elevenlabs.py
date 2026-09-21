"""ElevenLabsEngine: ElevenLabs text-to-speech over its REST API.

The voice is part of the URL path and the sample format is a query
parameter, so the body carries only the text and model. Requesting
``pcm_24000`` gives raw s16le back with no container to strip.
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


@dataclass(frozen=True)
class ElevenLabsConfig:
    voice_id: str
    model_id: str = "eleven_multilingual_v2"
    base_url: str = "https://api.elevenlabs.io"
    api_key_env: str = "ELEVENLABS_API_KEY"


@dataclass
class ElevenLabsEngine(TTSEngine):
    name: ClassVar[str] = "elevenlabs"

    config: ElevenLabsConfig
    api_key: str
    transport: httpx.BaseTransport | None = None
    sleep: Callable[[float], None] = time.sleep

    sample_rate: int = field(default=24000, init=False)
    max_concurrency: int = field(default=2, init=False)
    _client: httpx.Client = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._client = new_client(self.transport)

    def synthesize(self, texts: Sequence[str]) -> list[AudioClip]:
        url = f"{self.config.base_url.rstrip('/')}/v1/text-to-speech/{self.config.voice_id}"
        headers = {"xi-api-key": self.api_key}
        params = {"output_format": f"pcm_{self.sample_rate}"}
        clips = []
        for text in texts:
            response = post_with_retry(
                self._client,
                url,
                sleep=self.sleep,
                headers=headers,
                params=params,
                json={"text": text, "model_id": self.config.model_id},
            )
            clips.append(pcm_response_to_clip(response, self.sample_rate))
        return clips

    def fingerprint(self) -> str:
        return fingerprint_digest(self.name, self.config.voice_id, self.config.model_id)

    def close(self) -> None:
        self._client.close()
