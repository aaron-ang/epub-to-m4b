"""BreezeEngine: self-hosted Breeze-TTS sidecar, batched over its HTTP API.

Contract of the sidecar (see ``breeze_infer/api.py`` in the ``breeze-tts``
project): ``GET /health`` for readiness/sample rate, ``POST
/v1/audio/speech`` for a single non-batched clip (used here only to
bootstrap the one-time reference voice), and ``POST /v1/audio/speech/batch``
for everything else - texts as a JSON-encoded array, response body the
concatenation of every segment's s16le PCM with ``X-Segment-Bytes`` giving
each segment's length. The server serves one inference at a time and
answers 409 while busy.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import wave
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar

import httpx
import numpy as np

from epub_to_m4b.book import AudioClip
from epub_to_m4b.tts import guard
from epub_to_m4b.tts.base import TTSEngine, pcm16_to_float32
from epub_to_m4b.tts.http import TTSError
from epub_to_m4b.tts.sidecar import (
    DEFAULT_SIDECAR_POLICY,
    SidecarHandle,
    SidecarPolicy,
    post_until_free,
    start_or_adopt,
)

logger = logging.getLogger(__name__)

_DEFAULT_INSTRUCTION = "A clear, neutral adult narrator voice with a calm, steady reading pace."
# Fixed reference text read once to bootstrap a consistent narrator voice;
# without a reference, Breeze samples a new voice per call.
_REFERENCE_TEXT = "This is a clear, steady voice reading aloud for narration."
# Breeze's server uses triton, which needs ptxas on its path. A Breeze/CUDA
# detail, so it lives here rather than in the engine-agnostic tts/sidecar.py.
_SIDECAR_ENV = {"TRITON_PTXAS_PATH": "/usr/local/cuda/bin/ptxas"}


@dataclass(frozen=True)
class BreezeConfig:
    weights_dir: Path
    command: Sequence[str]
    cache_dir: Path
    # Any free unprivileged port; must match a server you want adopted.
    port: int = 7861
    instruction: str = _DEFAULT_INSTRUCTION
    # Classifier-free-guidance strength the model card recommends; higher
    # follows the instruction harder at the cost of naturalness.
    cfg_scale: float = 4.0
    # Fixed so re-synthesis is reproducible and cache keys stay valid.
    seed: int = 42
    # Texts per POST. Bounded by server VRAM; larger batches stop helping
    # once the GPU is saturated.
    batch_size: int = 64


def _write_wav(path: Path, pcm_bytes: bytes, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(np.dtype(np.int16).itemsize)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm_bytes)


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass
class BreezeEngine(TTSEngine):
    """Batches sentences to a local Breeze-TTS sidecar over HTTP.

    Bootstraps the sidecar (adopt/spawn, see ``tts/sidecar.py``) and the
    reference voice eagerly on construction, so a fully-built engine is
    always immediately ready to synthesize.
    """

    name: ClassVar[str] = "breeze"

    config: BreezeConfig
    transport: httpx.BaseTransport | None = None
    policy: SidecarPolicy = DEFAULT_SIDECAR_POLICY

    sample_rate: int = field(init=False)
    _sidecar: SidecarHandle = field(init=False, repr=False)
    _client: httpx.Client = field(init=False, repr=False)
    _reference_wav: Path = field(init=False, repr=False)
    _reference_text: str = field(init=False, repr=False)

    def __post_init__(self) -> None:
        # The orchestrator sizes each synthesize() call by max_batch; leaving
        # the ABC default of 1 would feed the GPU one sentence per request.
        self.max_batch = self.config.batch_size
        client_kwargs: dict[str, httpx.BaseTransport] = {}
        if self.transport is not None:
            client_kwargs["transport"] = self.transport
        self._client = httpx.Client(**client_kwargs)  # type: ignore[arg-type]

        log_path = self.config.cache_dir / f"breeze-server-{self.config.port}.log"
        # The sidecar's spawn command is engine-agnostic (see tts/sidecar.py) -
        # it only appends --host/--port. Breeze's server also takes the
        # weights directory as a required positional arg, so it goes on the
        # end of the configured command here, not inside sidecar.py.
        self._sidecar = start_or_adopt(
            [*self.config.command, str(self.config.weights_dir)],
            self.config.port,
            log_path=log_path,
            policy=self.policy,
            # A variable the user already exported wins over our default.
            env={k: v for k, v in _SIDECAR_ENV.items() if k not in os.environ},
            transport=self.transport,
        )
        self.sample_rate = self._sidecar.sample_rate
        try:
            self._reference_wav, self._reference_text = self._load_or_create_reference_voice()
        except BaseException:
            self._client.close()
            self._sidecar.close()
            raise

    def _load_or_create_reference_voice(self) -> tuple[Path, str]:
        ref_dir = self.config.cache_dir / "breeze"
        ref_wav = ref_dir / "reference_voice.wav"
        ref_txt = ref_dir / "reference_voice.txt"
        if ref_wav.exists() and ref_txt.exists():
            return ref_wav, ref_txt.read_text(encoding="utf-8")

        response = self._client.post(
            f"{self._sidecar.base_url}/v1/audio/speech",
            data={
                "text": _REFERENCE_TEXT,
                "instruction": self.config.instruction,
                "cfg_scale": self.config.cfg_scale,
                "seed": self.config.seed,
            },
            timeout=self.policy.request_timeout,
        )
        response.raise_for_status()
        _write_wav(ref_wav, response.content, self.sample_rate)
        ref_txt.write_text(_REFERENCE_TEXT, encoding="utf-8")
        return ref_wav, _REFERENCE_TEXT

    def synthesize(self, texts: Sequence[str]) -> list[AudioClip]:
        all_texts = list(texts)
        clips: list[AudioClip] = []
        for start in range(0, len(all_texts), self.config.batch_size):
            chunk = all_texts[start : start + self.config.batch_size]
            clips.extend(self._synthesize_chunk(chunk, seed=self.config.seed))

        def reseed(texts: Sequence[str], attempt: int) -> list[AudioClip]:
            # The runaway subset never exceeds one incoming batch, so it fits one POST.
            return self._synthesize_chunk(list(texts), seed=self.config.seed + attempt)

        guarded, notes = guard.apply_guard(clips, all_texts, reseed)
        for note in notes:
            logger.info("Breeze %s", note)
        return guarded

    def _synthesize_chunk(self, chunk: Sequence[str], *, seed: int) -> list[AudioClip]:
        data = {
            "texts": json.dumps(list(chunk)),
            "instruction": self.config.instruction,
            "cfg_scale": self.config.cfg_scale,
            "ref_text": self._reference_text,
            "seed": seed,
            "max_new_tokens": guard.max_new_tokens(chunk),
        }
        response = self._post_batch_with_retry(data)
        return self._split_segments(response, expected_count=len(chunk))

    def _post_batch_with_retry(self, data: dict[str, object]) -> httpx.Response:
        def send() -> httpx.Response:
            # Reopen per attempt: httpx consumes the file body on each POST.
            with self._reference_wav.open("rb") as ref_audio_file:
                return self._client.post(
                    f"{self._sidecar.base_url}/v1/audio/speech/batch",
                    data=data,
                    files={"ref_audio": ("reference_voice.wav", ref_audio_file, "audio/wav")},
                    timeout=self.policy.batch_timeout,
                )

        response = post_until_free(
            send,
            policy=self.policy,
            on_busy=lambda: logger.info(
                "Breeze server busy, waiting for the running inference to finish"
            ),
        )
        response.raise_for_status()
        return response

    def _split_segments(self, response: httpx.Response, *, expected_count: int) -> list[AudioClip]:
        header = response.headers.get("X-Segment-Bytes", "")
        if not header:
            raise TTSError("Breeze batch response is missing the X-Segment-Bytes header")
        sizes = [int(value) for value in header.split(",")]
        if len(sizes) != expected_count:
            raise TTSError(
                f"Breeze batch response returned {len(sizes)} segments for {expected_count} texts"
            )
        sample_rate = int(response.headers.get("X-Sample-Rate", self.sample_rate))
        body = response.content
        if sum(sizes) != len(body):
            raise TTSError(
                f"Breeze batch response segment sizes sum to {sum(sizes)} "
                f"but body is {len(body)} bytes"
            )
        clips = []
        offset = 0
        for size in sizes:
            raw = body[offset : offset + size]
            offset += size
            pcm = np.frombuffer(raw, dtype="<i2")
            clips.append(AudioClip(samples=pcm16_to_float32(pcm), sample_rate=sample_rate))
        return clips

    def fingerprint(self) -> str:
        digest = hashlib.sha256()
        parts = (
            self.name,
            str(self.config.weights_dir),
            self.config.instruction,
            repr(self.config.cfg_scale),
            repr(self.config.seed),
            _sha256_file(self._reference_wav),
        )
        for part in parts:
            digest.update(part.encode("utf-8"))
            digest.update(b"\0")
        return digest.hexdigest()

    def close(self) -> None:
        self._sidecar.close()
        self._client.close()
