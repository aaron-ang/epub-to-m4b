from __future__ import annotations

import email
import json
from email.message import Message
from pathlib import Path

import httpx
import numpy as np
import pytest

from epub_to_m4b.tts import guard
from epub_to_m4b.tts.breeze import BreezeConfig, BreezeEngine
from epub_to_m4b.tts.http import TTSError

_SAMPLE_RATE = 24000


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


def _health_response(sample_rate: int = _SAMPLE_RATE) -> httpx.Response:
    return httpx.Response(200, json={"status": "ok", "sample_rate": sample_rate})


def _config(tmp_path: Path, **overrides: object) -> BreezeConfig:
    defaults: dict[str, object] = {
        "weights_dir": tmp_path / "weights",
        "command": ["fake-breeze-command"],
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

    transport = httpx.MockTransport(handler)
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
            assert int(fields["max_new_tokens"]) > 0
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

    transport = httpx.MockTransport(handler)
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

    transport = httpx.MockTransport(handler)
    config = _config(tmp_path)
    engine = BreezeEngine(config, transport=transport, busy_retries=5, busy_wait_seconds=0.0)

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

    transport = httpx.MockTransport(handler)
    config = _config(tmp_path)
    engine = BreezeEngine(config, transport=transport, busy_retries=2, busy_wait_seconds=0.0)

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

    transport = httpx.MockTransport(handler)
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

    transport = httpx.MockTransport(handler)
    config = _config(tmp_path)
    engine = BreezeEngine(config, transport=transport)

    with pytest.raises(TTSError, match="2 segments"):
        engine.synthesize(["a", "b", "c"])
    engine.close()


def test_too_long_clip_gets_routed_through_guard(tmp_path: Path) -> None:
    text = "a runaway sentence"
    retry_limit = guard.retry_limit_seconds(text)
    runaway_seconds = retry_limit + 5.0
    recovered_seconds = retry_limit - 0.5
    batch_calls: list[tuple[list[str], int]] = []

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
            seed = int(fields["seed"])
            batch_calls.append((texts, seed))
            # the original batched call runs away; the reseed comes back short
            seconds = runaway_seconds if seed == config.seed else recovered_seconds
            segments = [_pcm_bytes(seconds) for _ in texts]
            return httpx.Response(
                200,
                content=b"".join(segments),
                headers={
                    "X-Segment-Bytes": ",".join(str(len(s)) for s in segments),
                    "X-Sample-Rate": str(_SAMPLE_RATE),
                },
            )
        raise AssertionError(f"unexpected path {request.url.path}")

    transport = httpx.MockTransport(handler)
    config = _config(tmp_path)
    engine = BreezeEngine(config, transport=transport)

    (clip,) = engine.synthesize([text])
    engine.close()

    assert clip.seconds <= retry_limit + 0.01
    assert batch_calls == [([text], config.seed), ([text], config.seed + 1)]


def test_several_runaways_reseed_as_one_post_per_attempt(tmp_path: Path) -> None:
    ok, run_a, run_b = "fine", "runaway alpha", "runaway beta"
    limit = min(guard.retry_limit_seconds(run_a), guard.retry_limit_seconds(run_b))
    batch_calls: list[tuple[list[str], int]] = []

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
            seed = int(fields["seed"])
            batch_calls.append((texts, seed))
            if seed == config.seed:
                segments = [_pcm_bytes(0.1 if t == ok else limit + 5.0) for t in texts]
            else:
                segments = [_pcm_bytes(limit - 0.5) for _ in texts]
            return httpx.Response(
                200,
                content=b"".join(segments),
                headers={
                    "X-Segment-Bytes": ",".join(str(len(s)) for s in segments),
                    "X-Sample-Rate": str(_SAMPLE_RATE),
                },
            )
        raise AssertionError(f"unexpected path {request.url.path}")

    transport = httpx.MockTransport(handler)
    config = _config(tmp_path)
    engine = BreezeEngine(config, transport=transport)

    clips = engine.synthesize([ok, run_a, run_b])
    engine.close()

    # one original POST, then exactly one reseed POST carrying both runaways
    assert batch_calls == [
        ([ok, run_a, run_b], config.seed),
        ([run_a, run_b], config.seed + 1),
    ]
    assert clips[0].seconds <= 0.11
    assert clips[1].seconds <= guard.retry_limit_seconds(run_a)
    assert clips[2].seconds <= guard.retry_limit_seconds(run_b)


def test_runaways_still_over_limit_get_second_attempt_post(tmp_path: Path) -> None:
    run_a, run_b = "runaway alpha", "runaway beta"
    limit = min(guard.retry_limit_seconds(run_a), guard.retry_limit_seconds(run_b))
    batch_calls: list[tuple[list[str], int]] = []

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
            seed = int(fields["seed"])
            batch_calls.append((texts, seed))
            if seed == config.seed + 1:
                # attempt 1: alpha recovers, beta still runs away
                segments = [_pcm_bytes(limit - 0.5 if t == run_a else limit + 4.0) for t in texts]
            elif seed == config.seed + 2:
                segments = [_pcm_bytes(limit - 0.5) for _ in texts]
            else:
                segments = [_pcm_bytes(limit + 5.0) for _ in texts]
            return httpx.Response(
                200,
                content=b"".join(segments),
                headers={
                    "X-Segment-Bytes": ",".join(str(len(s)) for s in segments),
                    "X-Sample-Rate": str(_SAMPLE_RATE),
                },
            )
        raise AssertionError(f"unexpected path {request.url.path}")

    transport = httpx.MockTransport(handler)
    config = _config(tmp_path)
    engine = BreezeEngine(config, transport=transport)

    clips = engine.synthesize([run_a, run_b])
    engine.close()

    assert batch_calls == [
        ([run_a, run_b], config.seed),
        ([run_a, run_b], config.seed + 1),
        ([run_b], config.seed + 2),
    ]
    assert all(
        c.seconds <= guard.retry_limit_seconds(t)
        for c, t in zip(clips, [run_a, run_b], strict=True)
    )


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
    transport_a = httpx.MockTransport(make_handler(_pcm_bytes(1.0)))
    engine_a1 = BreezeEngine(config_a, transport=transport_a)
    fp_a1 = engine_a1.fingerprint()
    engine_a1.close()

    engine_a2 = BreezeEngine(config_a, transport=transport_a)
    fp_a2 = engine_a2.fingerprint()
    engine_a2.close()
    assert fp_a1 == fp_a2

    config_b = _config(tmp_path / "b")
    transport_b = httpx.MockTransport(make_handler(_pcm_bytes(2.0)))  # different reference audio
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

    transport = httpx.MockTransport(handler)
    config = _config(tmp_path)
    engine = BreezeEngine(config, transport=transport)
    assert engine._sidecar.owned is False
    engine.close()  # must not raise
