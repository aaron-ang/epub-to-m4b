"""DramaEngine: Fish Audio Drama 3 on a local vLLM-Omni sidecar.

Contract of the sidecar (``vllm serve <model> --omni`` from vLLM-Omni):
``GET /health`` for readiness, ``GET /v1/models`` for the served model id,
and ``POST /v1/audio/speech`` with one text per request. The server batches
in-flight requests on the GPU itself (continuous batching) and queues the
rest rather than answering busy, so the GPU is kept full by sending many
requests at once: ``synthesize`` posts every text of its batch concurrently.

The reference voice is uploaded once (``POST /v1/audio/voices``, kept on
the server's disk) under a name derived from the audio and its transcript,
and every request names it as ``voice``: the server then encodes the
reference once and reuses the codes, instead of re-encoding an inline
``ref_audio`` per request. Each request carries a ``seed``, which the server
honours exactly: the same seed reproduces the same take, so retries reseed
with ``seed + round``. ``max_new_tokens`` counts codec frames.
"""

from __future__ import annotations

import hashlib
import io
import math
import mimetypes
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from http import HTTPMethod
from pathlib import Path
from typing import Any, ClassVar

import httpx
import soundfile as sf

from epub_to_m4b.book import AudioClip
from epub_to_m4b.tts import guard
from epub_to_m4b.tts.base import TTSEngine, fingerprint_digest
from epub_to_m4b.tts.guard import DEFAULT_RUNAWAY_POLICY, RunawayPolicy
from epub_to_m4b.tts.http import TTSError, pcm_response_to_clip, request_local
from epub_to_m4b.tts.sidecar import (
    SidecarHandle,
    SidecarPolicy,
    sidecar_client,
    start_or_adopt,
)

# Drama 3's codec decodes 44.1 kHz audio at 2048 samples per frame, so one
# max_new_tokens step is 2048 / 44100 s of speech.
_SAMPLE_RATE = 44100
_FRAME_RATE = _SAMPLE_RATE / 2048
# Every request's prompt holds the reference voice (one row per codec frame
# of its audio), its transcript and the text, at most one row per UTF-8 byte
# each (byte-level BPE), plus the chat template's own tokens, which this
# bounds. Prompt plus
# max_new_tokens must fit the server's max_model_len, or it answers 400.
_TEMPLATE_ROWS = 256
# vLLM-Omni's speech endpoint answers 400 to max_new_tokens above this
# (TTSModelAdapter.max_new_tokens_max), whatever room the context has left. No
# endpoint reports it, so it is fixed here.
_MAX_NEW_TOKENS = 4096
# max_chars counts characters but the prompt counts bytes, so it is sized
# for text that is all 4-byte UTF-8 characters, the most a character takes.
_MAX_UTF8_BYTES = 4
_WIDEST_CHAR = "\U0001f600"
# Speech requests resent after a reset connection. The same text and seed
# give the same clip, so a resend is safe; a dead server fails it too.
_SPEECH_RESENDS = 1
# Loading the ~38 GiB of weights, compiling and capturing CUDA graphs takes
# about ten minutes on a DGX Spark; a request waits behind the server's
# queue of up to max_concurrency * batch_size others.
DRAMA_SIDECAR_POLICY = SidecarPolicy(startup_timeout=1800.0, request_timeout=1800.0)


@dataclass(frozen=True)
class DramaConfig:
    # Starts the server, e.g. ["vllm", "serve", "fishaudio/drama-3", "--omni"];
    # the sidecar appends --host/--port.
    command: Sequence[str]
    # Audio the narrator voice is cloned from (WAV, FLAC or MP3, up to 60 s),
    # and its exact transcript.
    reference_audio: Path
    reference_text: str
    cache_dir: Path
    # Model id the server serves; requests and the clip cache key use it.
    model: str = "fishaudio/drama-3"
    # Any free unprivileged port; must match a server you want adopted.
    port: int = 8091
    # Fixed so re-synthesis is reproducible.
    seed: int = 42
    # Texts per synthesize() call, all posted at once. The server decodes up
    # to its max_num_seqs together and queues the rest.
    batch_size: int = 64


@dataclass
class DramaEngine(TTSEngine):
    """Posts each batch's sentences concurrently to a local Drama 3 server.

    Bootstraps the sidecar (adopt/spawn, see ``tts/sidecar.py``) and checks
    the served model eagerly on construction, so a fully-built engine is
    always immediately ready to synthesize.
    """

    name: ClassVar[str] = "drama"

    config: DramaConfig
    transport: httpx.BaseTransport | None = None
    policy: SidecarPolicy = DRAMA_SIDECAR_POLICY
    runaway: RunawayPolicy = DEFAULT_RUNAWAY_POLICY

    sample_rate: int = field(default=_SAMPLE_RATE, init=False)
    # Two batches in flight: the next one fills the GPU while the current
    # one's longest sentences finish.
    max_concurrency: int = field(default=2, init=False)
    _sidecar: SidecarHandle = field(init=False, repr=False)
    _client: httpx.Client = field(init=False, repr=False)
    _ref_bytes: bytes = field(init=False, repr=False)
    _ref_sha256: str = field(init=False, repr=False)
    _voice: str = field(init=False, repr=False)
    # Rows of max_model_len left once the reference voice and template are in.
    _budget_rows: int = field(init=False, repr=False)
    _ref_rows: int = field(init=False, repr=False)

    def __post_init__(self) -> None:
        try:
            self._ref_bytes = self.config.reference_audio.read_bytes()
        except OSError as exc:
            raise TTSError(
                f"cannot read Drama reference_audio {self.config.reference_audio}: {exc}"
            ) from exc

        self._ref_sha256 = hashlib.sha256(self._ref_bytes).hexdigest()
        # Content-addressed, so a changed reference or transcript is a new
        # upload rather than a stale voice reused under the same name.
        voice_digest = fingerprint_digest(self._ref_sha256, self.config.reference_text)
        self._voice = f"e2m-{voice_digest[:16]}"
        try:
            ref_seconds = sf.info(io.BytesIO(self._ref_bytes)).duration
        except (sf.LibsndfileError, RuntimeError) as exc:
            raise TTSError(
                f"cannot decode Drama reference_audio {self.config.reference_audio}: {exc}"
            ) from exc

        self._ref_rows = math.ceil(ref_seconds * _FRAME_RATE) + _rows(self.config.reference_text)
        if self.config.batch_size < 1:
            raise TTSError("[engine.drama].batch_size must be at least 1")

        self.max_batch = self.config.batch_size
        self.retries = self.runaway.retries

        log_path = self.config.cache_dir / f"drama-server-{self.config.port}.log"
        self._sidecar = start_or_adopt(
            self.config.command,
            self.config.port,
            log_path=log_path,
            policy=self.policy,
            transport=self.transport,
        )
        in_flight = self.max_concurrency * self.max_batch
        self._client = sidecar_client(
            self.policy,
            max_connections=in_flight,
            transport=self.transport,
        )
        try:
            max_model_len = self._check_model()
            self._budget_rows = max_model_len - self._ref_rows - _TEMPLATE_ROWS
            self.max_chars = self._longest_fitting_text(max_model_len)
            self._ensure_voice()
        except BaseException:
            self._client.close()
            self._sidecar.close()
            raise

    def _check_model(self) -> int:
        """The served model's max_model_len; raises unless ``config.model`` is served."""
        base_url = self._sidecar.base_url
        response = self._request(HTTPMethod.GET, f"{base_url}/v1/models")
        try:
            body = response.json()
        except ValueError:
            body = None
        data = body.get("data") if isinstance(body, dict) else None
        cards = [m for m in data if isinstance(m, dict)] if isinstance(data, list) else []
        served = [m.get("id") for m in cards]
        if response.status_code != httpx.codes.OK or self.config.model not in served:
            raise TTSError(
                f"Drama server at {base_url} serves {served or 'no models'}, not "
                f"{self.config.model!r}. Stop it, or set [engine.drama].model to match"
            )

        card = cards[served.index(self.config.model)]
        max_model_len = card.get("max_model_len")
        if isinstance(max_model_len, bool) or not isinstance(max_model_len, int):
            raise TTSError(f"Drama server at {base_url} did not report max_model_len")

        return max_model_len

    def _request(
        self,
        method: HTTPMethod,
        url: str,
        *,
        resends: int = 0,
        **kwargs: Any,
    ) -> httpx.Response:
        return request_local(
            self._client,
            method,
            url,
            timeout=self.policy.request_timeout,
            resends=resends,
            **kwargs,
        )

    def _rows_for(self, text: str) -> int:
        """Generation rows left for ``text`` after its prompt, at most what
        the server accepts as ``max_new_tokens``."""
        return min(self._budget_rows - _rows(text), _MAX_NEW_TOKENS)

    def _longest_fitting_text(self, max_model_len: int) -> int:
        """Longest text whose retry limit, in frames, fits beside its prompt,
        whatever characters it holds.

        Real speech stays under its retry limit, so such a text can always
        finish within the context; a longer one is cut before sending.
        Raises when not even one character fits.
        """

        def fits(chars: int) -> bool:
            text = _WIDEST_CHAR * chars
            frames = math.ceil(guard.retry_limit_seconds(text, self.runaway) * _FRAME_RATE)
            return frames <= self._rows_for(text)

        # Both bounds are linear in chars: solve each, take the smaller, then
        # step down past rounding. Each text byte is also a prompt row.
        base = self.runaway.base_seconds * _FRAME_RATE
        per_char = self.runaway.seconds_per_char * _FRAME_RATE
        chars = min(
            math.floor((self._budget_rows - base) / (per_char + _MAX_UTF8_BYTES)),
            math.floor((_MAX_NEW_TOKENS - base) / per_char),
        )
        while chars > 0 and not fits(chars):
            chars -= 1
        if chars < 1:
            raise TTSError(
                f"reference voice {self.config.reference_audio} leaves no room in the "
                f"server's {max_model_len}-row context; use a shorter clip"
            )

        return chars

    def _ensure_voice(self) -> None:
        url = f"{self._sidecar.base_url}/v1/audio/voices"
        listed = self._request(HTTPMethod.GET, url)
        try:
            body = listed.json()
        except ValueError:
            body = None
        uploaded = body.get("uploaded_voices") if isinstance(body, dict) else None
        if listed.status_code != httpx.codes.OK or not isinstance(uploaded, list):
            raise TTSError(f"GET {url} answered {listed.status_code}: {listed.text[:200]}")

        if any(isinstance(v, dict) and v.get("id") == self._voice for v in uploaded):
            return

        filename = self.config.reference_audio.name
        mime = mimetypes.guess_type(filename)[0] or "audio/wav"
        response = self._request(
            HTTPMethod.POST,
            url,
            data={
                "name": self._voice,
                "consent": "epub-to-m4b",
                "ref_text": self.config.reference_text,
            },
            files={"audio_sample": (filename, self._ref_bytes, mime)},
        )
        if not response.is_success:
            raise TTSError(
                f"uploading reference voice {self.config.reference_audio} failed with status "
                f"{response.status_code}: {response.text[:200]}"
            )

    def synthesize(self, texts: Sequence[str]) -> list[AudioClip]:
        return self._synthesize_all(texts, seed=self.config.seed, capped=True)

    def clip_miss(self, text: str, clip: AudioClip, *, capped: bool) -> float:
        return guard.clip_miss_seconds(clip, text, self.runaway, capped=capped)

    def resynthesize(
        self, texts: Sequence[str], retry_round: int, *, capped: bool
    ) -> list[AudioClip]:
        return self._synthesize_all(texts, seed=self.config.seed + retry_round, capped=capped)

    def _synthesize_all(self, texts: Sequence[str], *, seed: int, capped: bool) -> list[AudioClip]:
        if not texts:
            return []
        with ThreadPoolExecutor(max_workers=len(texts)) as executor:
            return list(executor.map(lambda text: self._speak(text, seed, capped), texts))

    def _speak(self, text: str, seed: int, capped: bool) -> AudioClip:
        # Requests are independent on the server, so each gets its own cap
        # rather than the longest text's; uncapped means all the room left.
        room = self._rows_for(text)
        tokens = (
            min(guard.max_new_tokens([text], self.runaway, _FRAME_RATE), room) if capped else room
        )
        url = f"{self._sidecar.base_url}/v1/audio/speech"
        body = {
            "model": self.config.model,
            "input": text,
            "voice": self._voice,
            "response_format": "pcm",
            "seed": seed,
            "max_new_tokens": tokens,
        }
        response = self._request(HTTPMethod.POST, url, resends=_SPEECH_RESENDS, json=body)
        if not response.is_success:
            raise TTSError(
                f"POST {url} failed with status {response.status_code}: {response.text[:200]}"
            )

        rate = response.headers.get("X-Audio-Sample-Rate")
        if rate is not None and rate.strip() != str(self.sample_rate):
            raise TTSError(f"Drama server returned {rate} Hz audio, expected {self.sample_rate}")

        return pcm_response_to_clip(response, self.sample_rate)

    def fingerprint(self) -> str:
        return fingerprint_digest(
            self.name,
            self.config.model,
            self.config.reference_text,
            repr(self.config.seed),
            self._ref_sha256,
        )

    def close(self) -> None:
        self._sidecar.close()
        self._client.close()


def _rows(text: str) -> int:
    """Upper bound on the prompt rows ``text`` tokenizes to: its UTF-8 bytes."""
    return len(text.encode("utf-8"))
