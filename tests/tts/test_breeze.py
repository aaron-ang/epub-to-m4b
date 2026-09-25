from __future__ import annotations

import email
import json
import logging
import subprocess
from collections.abc import Callable
from email.message import Message
from pathlib import Path
from unittest.mock import MagicMock

import httpx
import numpy as np
import pytest

from epub_to_m4b.book import AudioClip
from epub_to_m4b.tts import guard
from epub_to_m4b.tts.breeze import BreezeConfig, BreezeEngine
from epub_to_m4b.tts.http import TTSError
from epub_to_m4b.tts.sidecar import SidecarPolicy

_SAMPLE_RATE = 24000
_FRAME_RATE = 12.5


def _pcm_bytes(seconds: float, sample_rate: int = _SAMPLE_RATE, amplitude: int = 1000) -> bytes:
    n = max(1, int(seconds * sample_rate))
    samples = np.full(n, amplitude, dtype="<i2")
    return samples.tobytes()


def _parse_multipart(request: httpx.Request) -> dict[str, str]:
    """Decode a multipart/form-data request's text fields (skips file fields)."""
    content_type = request.headers["content-type"]
    body = request.read()
    raw = f"Content-Type: {content_type}\r\nMIME-Version: 1.0\r\n\r\n".encode() + body
    msg: Message = email.message_from_bytes(raw)
    fields: dict[str, str] = {}
    for part in msg.walk():
        if part.is_multipart():
            continue
        name = part.get_param("name", header="content-disposition")
        if name is None:
            continue
        if part.get_param("filename", header="content-disposition"):
            continue
        payload = part.get_payload(decode=True)
        fields[str(name)] = payload.decode("utf-8") if payload is not None else ""
    return fields


_MODEL_DIGEST = "ab" * 32


def _model_body(**overrides: object) -> dict[str, object]:
    body: dict[str, object] = {
        "frame_rate": _FRAME_RATE,
        "model_digest": _MODEL_DIGEST,
        "max_new_tokens": 1500,
        "max_batch_texts": 128,
    }
    body.update(overrides)
    return body


def _mock_transport(
    handler: Callable[[httpx.Request], httpx.Response], **model: object
) -> httpx.MockTransport:
    """``handler`` behind a ``breeze-tts-server`` that answers ``/v1/model``."""

    def with_model(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/model":
            return httpx.Response(200, json=_model_body(**model))
        return handler(request)

    return httpx.MockTransport(with_model)


def _health_response(sample_rate: int = _SAMPLE_RATE) -> httpx.Response:
    return httpx.Response(200, json={"status": "ok", "sample_rate": sample_rate})


def _retry_limit(text: str) -> float:
    return guard.retry_limit_seconds(text, guard.DEFAULT_RUNAWAY_POLICY)


def _config(tmp_path: Path, **overrides: object) -> BreezeConfig:
    defaults: dict[str, object] = {
        "command": ["fake-breeze-command", "some/model-repo"],
        "cache_dir": tmp_path / "cache",
        "port": 7861,
    }
    defaults.update(overrides)
    return BreezeConfig(**defaults)  # type: ignore[arg-type]


def test_reference_voice_created_on_first_use_and_reused_on_second(tmp_path: Path) -> None:
    reference_pcm = _pcm_bytes(1.0)
    speech_calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return _health_response()
        if request.url.path == "/v1/audio/speech":
            speech_calls["n"] += 1
            return httpx.Response(
                200,
                content=reference_pcm,
                headers={"X-Sample-Rate": str(_SAMPLE_RATE), "X-Sample-Format": "s16le"},
            )
        raise AssertionError(f"unexpected path {request.url.path}")

    transport = _mock_transport(handler)
    config = _config(tmp_path)

    engine1 = BreezeEngine(config, transport=transport)
    assert speech_calls["n"] == 1
    ref_wav = config.cache_dir / "breeze" / "reference_voice.wav"
    ref_txt = config.cache_dir / "breeze" / "reference_voice.txt"
    assert ref_wav.exists()
    expected_text = "This is a clear, steady voice reading aloud for narration."
    assert ref_txt.read_text(encoding="utf-8") == expected_text
    engine1.close()

    engine2 = BreezeEngine(config, transport=transport)
    assert speech_calls["n"] == 1  # not called again - reused from disk
    engine2.close()


def test_batch_splitting_respects_batch_size(tmp_path: Path) -> None:
    batch_calls: list[list[str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return _health_response()
        if request.url.path == "/v1/audio/speech":
            return httpx.Response(
                200,
                content=_pcm_bytes(1.0),
                headers={"X-Sample-Rate": str(_SAMPLE_RATE), "X-Sample-Format": "s16le"},
            )
        if request.url.path == "/v1/audio/speech/batch":
            fields = _parse_multipart(request)
            texts = json.loads(fields["texts"])
            batch_calls.append(texts)
            assert fields["instruction"] == config.instruction
            assert float(fields["cfg_scale"]) == config.cfg_scale
            assert (
                fields["ref_text"] == "This is a clear, steady voice reading aloud for narration."
            )
            assert int(fields["seed"]) == config.seed
            assert int(fields["max_new_tokens"]) == guard.max_new_tokens(
                texts, guard.DEFAULT_RUNAWAY_POLICY, _FRAME_RATE
            )
            segments = [_pcm_bytes(0.1) for _ in texts]
            return httpx.Response(
                200,
                content=b"".join(segments),
                headers={
                    "X-Segment-Bytes": ",".join(str(len(s)) for s in segments),
                    "X-Sample-Rate": str(_SAMPLE_RATE),
                },
            )
        raise AssertionError(f"unexpected path {request.url.path}")

    transport = _mock_transport(handler)
    config = _config(tmp_path, batch_size=3)
    engine = BreezeEngine(config, transport=transport)
    # The orchestrator only ever passes max_batch texts per call, so a stale
    # ABC default of 1 here silently degrades the GPU to one sentence per POST.
    assert engine.max_batch == 3

    texts = [f"sentence {i}" for i in range(7)]
    clips = engine.synthesize(texts)
    engine.close()

    assert len(clips) == 7
    assert [len(chunk) for chunk in batch_calls] == [3, 3, 1]
    assert [t for chunk in batch_calls for t in chunk] == texts


def test_409_triggers_wait_and_retry(tmp_path: Path) -> None:
    attempts = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return _health_response()
        if request.url.path == "/v1/audio/speech":
            return httpx.Response(
                200,
                content=_pcm_bytes(1.0),
                headers={"X-Sample-Rate": str(_SAMPLE_RATE), "X-Sample-Format": "s16le"},
            )
        if request.url.path == "/v1/audio/speech/batch":
            attempts["n"] += 1
            if attempts["n"] < 3:
                return httpx.Response(
                    409, json={"detail": "An inference request is already running."}
                )
            segments = [_pcm_bytes(0.1)]
            return httpx.Response(
                200,
                content=b"".join(segments),
                headers={
                    "X-Segment-Bytes": ",".join(str(len(s)) for s in segments),
                    "X-Sample-Rate": str(_SAMPLE_RATE),
                },
            )
        raise AssertionError(f"unexpected path {request.url.path}")

    transport = _mock_transport(handler)
    config = _config(tmp_path)
    engine = BreezeEngine(
        config, transport=transport, policy=SidecarPolicy(busy_timeout=0.005, busy_wait=0.001)
    )

    (clip,) = engine.synthesize(["one sentence"])
    engine.close()

    assert attempts["n"] == 3
    assert clip.sample_rate == _SAMPLE_RATE


def test_409_exhausts_retries_and_raises(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return _health_response()
        if request.url.path == "/v1/audio/speech":
            return httpx.Response(
                200,
                content=_pcm_bytes(1.0),
                headers={"X-Sample-Rate": str(_SAMPLE_RATE), "X-Sample-Format": "s16le"},
            )
        if request.url.path == "/v1/audio/speech/batch":
            return httpx.Response(409, json={"detail": "An inference request is already running."})
        raise AssertionError(f"unexpected path {request.url.path}")

    transport = _mock_transport(handler)
    config = _config(tmp_path)
    engine = BreezeEngine(
        config, transport=transport, policy=SidecarPolicy(busy_timeout=0.002, busy_wait=0.001)
    )

    with pytest.raises(httpx.HTTPStatusError):
        engine.synthesize(["one sentence"])
    engine.close()


def test_segment_bytes_header_parsing_splits_pcm_correctly(tmp_path: Path) -> None:
    segment_durations = [0.05, 0.2, 0.02]
    segments = [_pcm_bytes(d) for d in segment_durations]

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return _health_response()
        if request.url.path == "/v1/audio/speech":
            return httpx.Response(
                200,
                content=_pcm_bytes(1.0),
                headers={"X-Sample-Rate": str(_SAMPLE_RATE), "X-Sample-Format": "s16le"},
            )
        if request.url.path == "/v1/audio/speech/batch":
            return httpx.Response(
                200,
                content=b"".join(segments),
                headers={
                    "X-Segment-Bytes": ",".join(str(len(s)) for s in segments),
                    "X-Sample-Rate": str(_SAMPLE_RATE),
                },
            )
        raise AssertionError(f"unexpected path {request.url.path}")

    transport = _mock_transport(handler)
    config = _config(tmp_path)
    engine = BreezeEngine(config, transport=transport)

    clips = engine.synthesize(["a", "b", "c"])
    engine.close()

    assert len(clips) == 3
    for clip, expected_bytes in zip(clips, segments, strict=True):
        assert len(clip.samples) == len(expected_bytes) // 2
        assert clip.sample_rate == _SAMPLE_RATE
        expected_floats = np.frombuffer(expected_bytes, dtype="<i2").astype(np.float32) / 32768.0
        np.testing.assert_array_equal(clip.samples, expected_floats)


def test_segment_count_mismatch_raises_clear_error(tmp_path: Path) -> None:
    # Server returns only 2 segments for 3 requested texts - a precise error
    # should point at the count mismatch, not surface later as a generic
    # zip() ValueError once the guard tries to pair clips with texts.
    segments = [_pcm_bytes(0.1), _pcm_bytes(0.1)]

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return _health_response()
        if request.url.path == "/v1/audio/speech":
            return httpx.Response(
                200,
                content=_pcm_bytes(1.0),
                headers={"X-Sample-Rate": str(_SAMPLE_RATE), "X-Sample-Format": "s16le"},
            )
        if request.url.path == "/v1/audio/speech/batch":
            return httpx.Response(
                200,
                content=b"".join(segments),
                headers={
                    "X-Segment-Bytes": ",".join(str(len(s)) for s in segments),
                    "X-Sample-Rate": str(_SAMPLE_RATE),
                },
            )
        raise AssertionError(f"unexpected path {request.url.path}")

    transport = _mock_transport(handler)
    config = _config(tmp_path)
    engine = BreezeEngine(config, transport=transport)

    with pytest.raises(TTSError, match="2 segments"):
        engine.synthesize(["a", "b", "c"])
    engine.close()


def test_fingerprint_is_stable_and_changes_with_reference_voice(tmp_path: Path) -> None:
    def make_handler(reference_pcm: bytes) -> object:
        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/health":
                return _health_response()
            if request.url.path == "/v1/audio/speech":
                return httpx.Response(
                    200,
                    content=reference_pcm,
                    headers={"X-Sample-Rate": str(_SAMPLE_RATE), "X-Sample-Format": "s16le"},
                )
            raise AssertionError(f"unexpected path {request.url.path}")

        return handler

    config_a = _config(tmp_path / "a")
    transport_a = _mock_transport(make_handler(_pcm_bytes(1.0)))
    engine_a1 = BreezeEngine(config_a, transport=transport_a)
    fp_a1 = engine_a1.fingerprint()
    engine_a1.close()

    engine_a2 = BreezeEngine(config_a, transport=transport_a)
    fp_a2 = engine_a2.fingerprint()
    engine_a2.close()
    assert fp_a1 == fp_a2

    config_b = _config(tmp_path / "b")
    transport_b = _mock_transport(make_handler(_pcm_bytes(2.0)))  # different reference audio
    engine_b = BreezeEngine(config_b, transport=transport_b)
    fp_b = engine_b.fingerprint()
    engine_b.close()

    assert fp_a1 != fp_b


def test_close_stops_owned_sidecar_but_not_adopted(tmp_path: Path) -> None:
    # Health always healthy => sidecar is adopted, not spawned; close() must
    # not attempt to kill anything (there's no process to kill).
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return _health_response()
        if request.url.path == "/v1/audio/speech":
            return httpx.Response(
                200,
                content=_pcm_bytes(1.0),
                headers={"X-Sample-Rate": str(_SAMPLE_RATE), "X-Sample-Format": "s16le"},
            )
        raise AssertionError(f"unexpected path {request.url.path}")

    transport = _mock_transport(handler)
    config = _config(tmp_path)
    engine = BreezeEngine(config, transport=transport)
    assert engine._sidecar.owned is False
    engine.close()  # must not raise


def _spawning_transport() -> httpx.MockTransport:
    """First /health refuses (nothing listening -> spawn), then healthy."""
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            calls["n"] += 1
            if calls["n"] == 1:
                raise httpx.ConnectError("connection refused")
            return _health_response()
        if request.url.path == "/v1/audio/speech":
            return httpx.Response(
                200,
                content=_pcm_bytes(1.0),
                headers={"X-Sample-Rate": str(_SAMPLE_RATE), "X-Sample-Format": "s16le"},
            )
        raise AssertionError(f"unexpected path {request.url.path}")

    return _mock_transport(handler)


def test_spawn_sets_default_triton_ptxas_path_when_unset(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("TRITON_PTXAS_PATH", raising=False)
    fake_process = MagicMock(spec=subprocess.Popen)
    fake_process.poll.return_value = None
    popen = MagicMock(return_value=fake_process)
    monkeypatch.setattr(subprocess, "Popen", popen)

    engine = BreezeEngine(_config(tmp_path), transport=_spawning_transport())
    engine.close()

    env = popen.call_args.kwargs["env"]
    assert env["TRITON_PTXAS_PATH"] == "/usr/local/cuda/bin/ptxas"


def test_spawn_respects_already_set_triton_ptxas_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("TRITON_PTXAS_PATH", "/custom/ptxas")
    fake_process = MagicMock(spec=subprocess.Popen)
    fake_process.poll.return_value = None
    popen = MagicMock(return_value=fake_process)
    monkeypatch.setattr(subprocess, "Popen", popen)

    engine = BreezeEngine(_config(tmp_path), transport=_spawning_transport())
    engine.close()

    env = popen.call_args.kwargs["env"]
    assert env["TRITON_PTXAS_PATH"] == "/custom/ptxas"


def _ready_transport(
    batch_fields: list[dict[str, str]] | None = None, **model: object
) -> httpx.MockTransport:
    """An adopted, healthy sidecar that answers every batch text with a short clip."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return _health_response()
        if request.url.path == "/v1/audio/speech":
            return httpx.Response(
                200,
                content=_pcm_bytes(1.0),
                headers={"X-Sample-Rate": str(_SAMPLE_RATE), "X-Sample-Format": "s16le"},
            )
        if request.url.path == "/v1/audio/speech/batch":
            fields = _parse_multipart(request)
            if batch_fields is not None:
                batch_fields.append(fields)
            segments = [_pcm_bytes(0.1) for _ in json.loads(fields["texts"])]
            return httpx.Response(
                200,
                content=b"".join(segments),
                headers={
                    "X-Segment-Bytes": ",".join(str(len(s)) for s in segments),
                    "X-Sample-Rate": str(_SAMPLE_RATE),
                },
            )
        raise AssertionError(f"unexpected path {request.url.path}")

    return _mock_transport(handler, **model)


def test_token_cap_uses_frame_rate_from_server(tmp_path: Path) -> None:
    policy = guard.RunawayPolicy(cap_slack=2.0)
    batch_fields: list[dict[str, str]] = []
    transport = _ready_transport(batch_fields, frame_rate=25.0)
    engine = BreezeEngine(_config(tmp_path), transport=transport, runaway=policy)
    assert engine.server.frame_rate == 25.0
    engine.synthesize(["one sentence"])
    engine.close()

    (fields,) = batch_fields
    assert int(fields["max_new_tokens"]) == guard.max_new_tokens(["one sentence"], policy, 25.0)


def test_token_cap_clamped_to_server_max_new_tokens(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    texts = ["a" * 100, "b" * 100]
    wanted = guard.max_new_tokens(texts[:1], guard.DEFAULT_RUNAWAY_POLICY, _FRAME_RATE)
    server_max = wanted // 2
    batch_fields: list[dict[str, str]] = []
    transport = _ready_transport(batch_fields, max_new_tokens=server_max)
    # The fake server's clips are short for 100 chars; keep the short check out of it.
    no_short = guard.RunawayPolicy(min_chars=1000)
    engine = BreezeEngine(_config(tmp_path, batch_size=1), transport=transport, runaway=no_short)
    with caplog.at_level(logging.DEBUG, logger="epub_to_m4b.tts.breeze"):
        engine.synthesize(texts)
    engine.close()

    assert [int(f["max_new_tokens"]) for f in batch_fields] == [server_max, server_max]
    clamp_logs = [r for r in caplog.records if "clamped" in r.getMessage()]
    assert len(clamp_logs) == 1
    assert clamp_logs[0].levelno == logging.DEBUG


def test_synthesize_is_one_capped_first_pass_without_retries(tmp_path: Path) -> None:
    texts = ["a runaway sentence", "another one"]
    batch_fields: list[dict[str, str]] = []
    config = _config(tmp_path)
    engine = BreezeEngine(config, transport=_ready_transport(batch_fields))
    engine.synthesize(texts)
    engine.close()

    (fields,) = batch_fields
    assert json.loads(fields["texts"]) == texts
    assert int(fields["seed"]) == config.seed
    assert int(fields["max_new_tokens"]) == guard.max_new_tokens(
        texts, guard.DEFAULT_RUNAWAY_POLICY, _FRAME_RATE
    )


def test_resynthesize_capped_keeps_first_pass_cap(tmp_path: Path) -> None:
    texts = ["x" * 100]
    batch_fields: list[dict[str, str]] = []
    config = _config(tmp_path)
    engine = BreezeEngine(config, transport=_ready_transport(batch_fields))
    engine.resynthesize(texts, 1, capped=True)
    engine.close()

    (fields,) = batch_fields
    assert int(fields["seed"]) == config.seed + 1
    assert int(fields["max_new_tokens"]) == guard.max_new_tokens(
        texts, guard.DEFAULT_RUNAWAY_POLICY, _FRAME_RATE
    )


def test_resynthesize_uncapped_draws_seed_per_round(tmp_path: Path) -> None:
    texts = ["x" * 100, "y" * 3]
    batch_fields: list[dict[str, str]] = []
    config = _config(tmp_path, batch_size=1)
    engine = BreezeEngine(config, transport=_ready_transport(batch_fields))
    clips = engine.resynthesize(texts, 3, capped=False)
    engine.close()

    assert len(clips) == 2
    assert [json.loads(f["texts"]) for f in batch_fields] == [[texts[0]], [texts[1]]]
    assert {int(f["seed"]) for f in batch_fields} == {config.seed + 3}
    assert {int(f["max_new_tokens"]) for f in batch_fields} == {1500}


def test_retry_hooks_follow_runaway_policy(tmp_path: Path) -> None:
    policy = guard.RunawayPolicy(retries=5)
    engine = BreezeEngine(_config(tmp_path), transport=_ready_transport(), runaway=policy)
    text = "a runaway sentence"
    over = AudioClip(
        samples=np.zeros(int((_retry_limit(text) + 1.0) * _SAMPLE_RATE), dtype=np.float32),
        sample_rate=_SAMPLE_RATE,
    )
    engine.close()

    assert engine.retries == 5
    assert engine.clip_miss(text, over, capped=True) == float("inf")
    assert engine.clip_miss(text, over, capped=False) == pytest.approx(1.0, abs=1e-3)


def test_batch_size_clamped_to_server_max_batch_texts(tmp_path: Path) -> None:
    batch_fields: list[dict[str, str]] = []
    transport = _ready_transport(batch_fields, max_batch_texts=2)
    engine = BreezeEngine(_config(tmp_path, batch_size=64), transport=transport)
    assert engine.max_batch == 2
    engine.synthesize([f"sentence {i}" for i in range(5)])
    engine.close()
    assert [len(json.loads(f["texts"])) for f in batch_fields] == [2, 2, 1]


def test_server_without_model_route_raises_clear_error(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return _health_response()
        return httpx.Response(404, json={"detail": "Not Found"})

    with pytest.raises(TTSError, match=r"/v1/model answered 404.*breeze-tts-server"):
        BreezeEngine(_config(tmp_path), transport=httpx.MockTransport(handler))


def test_model_route_missing_fields_names_them(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return _health_response()
        if request.url.path == "/v1/model":
            return httpx.Response(200, json={})
        raise AssertionError(f"unexpected path {request.url.path}")

    with pytest.raises(TTSError, match="breeze-tts-server") as exc_info:
        BreezeEngine(_config(tmp_path), transport=httpx.MockTransport(handler))
    for key in ("frame_rate", "model_digest", "max_new_tokens", "max_batch_texts"):
        assert key in str(exc_info.value)


def test_model_route_non_object_body_raises(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return _health_response()
        return httpx.Response(200, json=[1, 2])

    with pytest.raises(TTSError, match="JSON object"):
        BreezeEngine(_config(tmp_path), transport=httpx.MockTransport(handler))


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("frame_rate", 0),
        ("frame_rate", "12.5"),
        ("frame_rate", True),
        ("model_digest", ""),
        ("max_new_tokens", 1.5),
        ("max_batch_texts", 0),
    ],
)
def test_invalid_model_field_raises(tmp_path: Path, key: str, value: object) -> None:
    with pytest.raises(TTSError, match=rf"{key}=.*breeze-tts-server"):
        BreezeEngine(_config(tmp_path), transport=_ready_transport(**{key: value}))


def _fingerprint(config: BreezeConfig, **model: object) -> str:
    engine = BreezeEngine(config, transport=_ready_transport(**model))
    try:
        return engine.fingerprint()
    finally:
        engine.close()


def test_fingerprint_follows_server_model_digest(tmp_path: Path) -> None:
    config = _config(tmp_path)
    fp = _fingerprint(config)
    assert _fingerprint(config) == fp
    assert _fingerprint(config, model_digest="cd" * 32) != fp


def test_fingerprint_ignores_model_path_in_command(tmp_path: Path) -> None:
    local = _config(tmp_path, command=["breeze-infer-api", "/models/breeze-tts-2"])
    moved = _config(tmp_path, command=["breeze-infer-api", "/elsewhere/breeze-tts-2"])
    repo = _config(tmp_path, command=["breeze-infer-api", "BreezeBlue/Breeze-TTS-2"])
    assert _fingerprint(local) == _fingerprint(moved) == _fingerprint(repo)
