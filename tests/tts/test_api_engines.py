"""Request shape, response decoding, fingerprint scope and retry wiring for
the hosted HTTP API engines, checked against a mock transport."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass

import httpx
import numpy as np
import pytest

from epub_to_m4b.tts.base import TTSEngine
from epub_to_m4b.tts.elevenlabs import ElevenLabsConfig, ElevenLabsEngine
from epub_to_m4b.tts.openai_compat import OpenAIConfig, OpenAIEngine

_KEY = "sk-test-key"


def _pcm(n_samples: int) -> bytes:
    return np.full(n_samples, 1000, dtype="<i2").tobytes()


@dataclass(frozen=True)
class _Case:
    """One engine's expected wire format plus knobs for the fingerprint tests."""

    make: Callable[..., TTSEngine]
    url: str
    method: str
    headers: dict[str, str]
    body: Callable[[str], dict[str, object]]
    audio_override: dict[str, object]
    base_url_override: dict[str, object]


def _openai(
    transport: httpx.BaseTransport | None = None,
    *,
    api_key: str = _KEY,
    sleep: Callable[[float], None] = lambda _s: None,
    **overrides: object,
) -> OpenAIEngine:
    fields: dict[str, object] = {
        "base_url": "https://api.openai.com/",
        "model": "gpt-4o-mini-tts",
        "voice": "alloy",
    }
    fields.update(overrides)
    config = OpenAIConfig(**fields)  # type: ignore[arg-type]
    return OpenAIEngine(config, api_key=api_key, transport=transport, sleep=sleep)


def _elevenlabs(
    transport: httpx.BaseTransport | None = None,
    *,
    api_key: str = _KEY,
    sleep: Callable[[float], None] = lambda _s: None,
    **overrides: object,
) -> ElevenLabsEngine:
    fields: dict[str, object] = {"voice_id": "voice123"}
    fields.update(overrides)
    config = ElevenLabsConfig(**fields)  # type: ignore[arg-type]
    return ElevenLabsEngine(config, api_key=api_key, transport=transport, sleep=sleep)


CASES = {
    "elevenlabs": _Case(
        make=_elevenlabs,
        url="https://api.elevenlabs.io/v1/text-to-speech/voice123?output_format=pcm_24000",
        method="POST",
        headers={"xi-api-key": _KEY},
        body=lambda text: {"text": text, "model_id": "eleven_multilingual_v2"},
        audio_override={"model_id": "eleven_turbo_v2_5"},
        base_url_override={"base_url": "https://proxy.example"},
    ),
    "openai": _Case(
        make=_openai,
        url="https://api.openai.com/v1/audio/speech",
        method="POST",
        headers={"Authorization": f"Bearer {_KEY}"},
        body=lambda text: {
            "model": "gpt-4o-mini-tts",
            "input": text,
            "voice": "alloy",
            "speed": 1.0,
            "response_format": "pcm",
        },
        audio_override={"voice": "nova"},
        base_url_override={"base_url": "https://proxy.example"},
    ),
}


@pytest.fixture(params=sorted(CASES))
def case(request: pytest.FixtureRequest) -> _Case:
    return CASES[str(request.param)]


def _recording_transport(
    sizes_by_text: dict[str, int],
) -> tuple[httpx.MockTransport, list[httpx.Request]]:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        body = json.loads(request.read())
        text = body.get("input") or body.get("text")
        return httpx.Response(200, content=_pcm(sizes_by_text[text]))

    return httpx.MockTransport(handler), requests


def test_request_shape(case: _Case) -> None:
    transport, requests = _recording_transport({"Hello there.": 4})
    with case.make(transport) as engine:
        engine.synthesize(["Hello there."])
    (request,) = requests
    assert request.method == case.method
    assert str(request.url) == case.url
    for name, value in case.headers.items():
        assert request.headers[name] == value
    assert json.loads(request.read()) == case.body("Hello there.")


def test_clips_come_back_in_input_order(case: _Case) -> None:
    texts = ["one", "two words", "three whole words"]
    sizes = {"one": 10, "two words": 20, "three whole words": 30}
    transport, requests = _recording_transport(sizes)
    with case.make(transport) as engine:
        clips = engine.synthesize(texts)
    assert len(requests) == len(texts)
    assert [len(clip.samples) for clip in clips] == [10, 20, 30]
    for clip in clips:
        assert clip.samples.dtype == np.float32
        assert clip.sample_rate == engine.sample_rate


def test_fingerprint_is_stable_and_tracks_only_audio_parameters(case: _Case) -> None:
    base = case.make().fingerprint()
    assert base == case.make().fingerprint()
    assert case.make(**case.audio_override).fingerprint() != base
    assert case.make(api_key="other-key").fingerprint() == base
    assert case.make(**case.base_url_override).fingerprint() == base


def test_close_closes_the_client(case: _Case) -> None:
    engine = case.make()
    engine.close()
    with pytest.raises(RuntimeError, match="closed"):
        engine.synthesize(["after close"])


def test_transient_5xx_is_retried_then_decoded(case: _Case) -> None:
    statuses = iter([503, 200])
    sleeps: list[float] = []

    def handler(_request: httpx.Request) -> httpx.Response:
        status = next(statuses)
        return httpx.Response(status, content=_pcm(8) if status == 200 else b"busy")

    with case.make(httpx.MockTransport(handler), sleep=sleeps.append) as engine:
        (clip,) = engine.synthesize(["retry me"])
    assert len(clip.samples) == 8
    assert sleeps == [1]


def test_engine_never_reads_the_environment(case: _Case, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "from-env")
    transport, requests = _recording_transport({"x": 2})
    with case.make(transport) as engine:
        engine.synthesize(["x"])
    (request,) = requests
    assert "from-env" not in "".join(request.headers.values())
