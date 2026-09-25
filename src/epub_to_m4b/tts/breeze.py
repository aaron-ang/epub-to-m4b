"""BreezeEngine: self-hosted Breeze-TTS sidecar, batched over its HTTP API.

Contract of the sidecar (upstream ``breeze_infer/api.py`` in the
``breeze-tts`` project, plus ``GET /v1/model`` for model facts added by its
``breeze-tts-server`` wrapper): ``GET /health`` for readiness/sample rate, ``POST
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
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar

import httpx
import numpy as np

from epub_to_m4b.book import AudioClip
from epub_to_m4b.tts import guard
from epub_to_m4b.tts.base import TTSEngine, fingerprint_digest, pcm16_to_float32
from epub_to_m4b.tts.guard import DEFAULT_RUNAWAY_POLICY, RunawayPolicy
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
    # Starts the server; ends with the model (a local dir or an HF repo id,
    # resolved by breeze-tts). The sidecar appends --host/--port.
    command: Sequence[str]
    cache_dir: Path
    # Any free unprivileged port; must match a server you want adopted.
    port: int = 7861
    instruction: str = _DEFAULT_INSTRUCTION
    # Classifier-free-guidance strength the model card recommends; higher
    # follows the instruction harder at the cost of naturalness.
    cfg_scale: float = 4.0
    # Fixed so re-synthesis is reproducible.
    seed: int = 42
    # Texts per POST, clamped to the server's max_batch_texts. Bounded by
    # server VRAM; larger batches stop helping once the GPU is saturated.
    batch_size: int = 64


@dataclass(frozen=True)
class ServerInfo:
    """Model facts the Breeze server reports on ``GET /v1/model``.

    The server owns the model, so it is the source of truth for the codec
    frame rate (the unit ``max_new_tokens`` counts in), the model identity
    that partitions the clip cache, and its own request limits.
    """

    frame_rate: float
    model_digest: str
    max_new_tokens: int
    max_batch_texts: int

    @classmethod
    def from_model_body(cls, body: Mapping[str, object], base_url: str) -> ServerInfo:
        """Parse a ``/v1/model`` body; a missing or invalid field is a wrong server."""
        frame_rate = _positive_number(body.get("frame_rate"))
        model_digest = _nonempty_str(body.get("model_digest"))
        max_new_tokens = _positive_int(body.get("max_new_tokens"))
        max_batch_texts = _positive_int(body.get("max_batch_texts"))
        if (
            frame_rate is None
            or model_digest is None
            or max_new_tokens is None
            or max_batch_texts is None
        ):
            parsed = {
                "frame_rate": frame_rate,
                "model_digest": model_digest,
                "max_new_tokens": max_new_tokens,
                "max_batch_texts": max_batch_texts,
            }
            bad = ", ".join(f"{k}={body.get(k)!r}" for k, v in parsed.items() if v is None)
            raise _wrong_server(base_url, f"missing or invalid /v1/model field(s): {bad}")
        return cls(frame_rate, model_digest, max_new_tokens, max_batch_texts)


def _wrong_server(base_url: str, problem: str) -> TTSError:
    return TTSError(
        f"Breeze server at {base_url}: {problem}. Stop it and start it with "
        "`breeze-tts-server` from the current breeze-tts"
    )


def _positive_number(value: object) -> float | None:
    if isinstance(value, int | float) and not isinstance(value, bool) and value > 0:
        return float(value)
    return None


def _positive_int(value: object) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool) and value > 0:
        return value
    return None


def _nonempty_str(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


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
    runaway: RunawayPolicy = DEFAULT_RUNAWAY_POLICY

    sample_rate: int = field(init=False)
    server: ServerInfo = field(init=False)
    _cap_clamp_logged: bool = field(default=False, init=False, repr=False)
    _sidecar: SidecarHandle = field(init=False, repr=False)
    _client: httpx.Client = field(init=False, repr=False)
    _reference_wav: Path = field(init=False, repr=False)
    _reference_text: str = field(init=False, repr=False)

    def __post_init__(self) -> None:
        client_kwargs: dict[str, httpx.BaseTransport] = {}
        if self.transport is not None:
            client_kwargs["transport"] = self.transport
        self._client = httpx.Client(**client_kwargs)  # type: ignore[arg-type]

        log_path = self.config.cache_dir / f"breeze-server-{self.config.port}.log"
        self._sidecar = start_or_adopt(
            self.config.command,
            self.config.port,
            log_path=log_path,
            policy=self.policy,
            # A variable the user already exported wins over our default.
            env={k: v for k, v in _SIDECAR_ENV.items() if k not in os.environ},
            transport=self.transport,
        )
        self.sample_rate = self._sidecar.sample_rate
        try:
            self.server = self._fetch_server_info()
            # The orchestrator sizes each synthesize() call by max_batch; leaving
            # the ABC default of 1 would feed the GPU one sentence per request.
            self.max_batch = min(self.config.batch_size, self.server.max_batch_texts)
            self.retries = self.runaway.retries
            self._reference_wav, self._reference_text = self._load_or_create_reference_voice()
        except BaseException:
            self._client.close()
            self._sidecar.close()
            raise

    def _fetch_server_info(self) -> ServerInfo:
        base_url = self._sidecar.base_url
        response = self._client.get(f"{base_url}/v1/model", timeout=self.policy.request_timeout)
        if response.status_code != httpx.codes.OK:
            raise _wrong_server(base_url, f"GET /v1/model answered {response.status_code}")
        try:
            body = response.json()
        except ValueError:
            body = None
        if not isinstance(body, dict):
            raise _wrong_server(base_url, "GET /v1/model did not return a JSON object")
        return ServerInfo.from_model_body(body, base_url)

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
        return self._synthesize_all(texts, seed=self.config.seed, capped=True)

    def clip_miss(self, text: str, clip: AudioClip, *, capped: bool) -> float:
        return guard.clip_miss_seconds(clip, text, self.runaway, capped=capped)

    def resynthesize(
        self, texts: Sequence[str], retry_round: int, *, capped: bool
    ) -> list[AudioClip]:
        return self._synthesize_all(texts, seed=self.config.seed + retry_round, capped=capped)

    def _synthesize_all(self, texts: Sequence[str], *, seed: int, capped: bool) -> list[AudioClip]:
        all_texts = list(texts)
        clips: list[AudioClip] = []
        for start in range(0, len(all_texts), self.max_batch):
            chunk = all_texts[start : start + self.max_batch]
            tokens = self._max_new_tokens(chunk) if capped else self.server.max_new_tokens
            clips.extend(self._synthesize_chunk(chunk, seed=seed, max_new_tokens=tokens))
        return clips

    def _synthesize_chunk(
        self, chunk: Sequence[str], *, seed: int, max_new_tokens: int
    ) -> list[AudioClip]:
        data = {
            "texts": json.dumps(list(chunk)),
            "instruction": self.config.instruction,
            "cfg_scale": self.config.cfg_scale,
            "ref_text": self._reference_text,
            "seed": seed,
            "max_new_tokens": max_new_tokens,
        }
        response = self._post_batch_with_retry(data)
        return self._split_segments(response, expected_count=len(chunk))

    def _max_new_tokens(self, chunk: Sequence[str]) -> int:
        tokens = guard.max_new_tokens(chunk, self.runaway, self.server.frame_rate)
        if tokens <= self.server.max_new_tokens:
            return tokens
        if not self._cap_clamp_logged:
            logger.debug(
                "Breeze token cap %d clamped to the server's max_new_tokens %d",
                tokens,
                self.server.max_new_tokens,
            )
            self._cap_clamp_logged = True
        return self.server.max_new_tokens

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
        return fingerprint_digest(
            self.name,
            self.server.model_digest,
            self.config.instruction,
            repr(self.config.cfg_scale),
            repr(self.config.seed),
            _sha256_file(self._reference_wav),
        )

    def close(self) -> None:
        self._sidecar.close()
        self._client.close()
