from __future__ import annotations

import json
import math
import threading
from collections.abc import Callable
from pathlib import Path

import httpx
import numpy as np
import pytest
import soundfile as sf

from epub_to_m4b.book import AudioClip
from epub_to_m4b.tts import guard
from epub_to_m4b.tts.drama import DramaConfig, DramaEngine
from epub_to_m4b.tts.http import TTSError

_SAMPLE_RATE = 44100
_FRAME_RATE = _SAMPLE_RATE / 2048
_MODEL = "fishaudio/drama-3"
_MAX_MODEL_LEN = 8192
_MAX_NEW_TOKENS = 4096
# A context whose room after any test prompt stays under _MAX_NEW_TOKENS.
_SMALL_MODEL_LEN = 2048
_REF_SECONDS = 2.0


def _wav(path: Path, seconds: float = _REF_SECONDS, level: float = 0.1) -> None:
    sf.write(path, np.full(int(seconds * _SAMPLE_RATE), level, dtype=np.float32), _SAMPLE_RATE)


def _pcm(seconds: float) -> bytes:
    return np.full(int(seconds * _SAMPLE_RATE), 1000, dtype="<i2").tobytes()


def _config(tmp_path: Path, **overrides: object) -> DramaConfig:
    ref = tmp_path / "voice.wav"
    if not ref.exists():
        _wav(ref)
    values: dict[str, object] = {
        "command": ["vllm", "serve", _MODEL, "--omni"],
        "reference_audio": ref,
        "reference_text": "Reference words.",
        "cache_dir": tmp_path / "cache",
    }
    values.update(overrides)
    return DramaConfig(**values)  # type: ignore[arg-type]


def _transport(
    speech: Callable[[dict[str, object]], httpx.Response] | None = None,
    models: list[str] | None = None,
    voices: list[str] | None = None,
    uploads: list[bytes] | None = None,
    max_model_len: object = _MAX_MODEL_LEN,
) -> httpx.MockTransport:
    """An adopted, healthy server; ``speech`` answers each parsed speech body.

    ``voices`` holds the uploaded voice ids, shared with the test; every
    upload body lands in ``uploads``.
    """
    voices = [] if voices is None else voices

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "healthy"})
        if request.url.path == "/v1/models":
            ids = [_MODEL] if models is None else models
            cards = [{"id": i, "max_model_len": max_model_len} for i in ids]
            return httpx.Response(200, json={"object": "list", "data": cards})
        if request.url.path == "/v1/audio/voices" and request.method == "GET":
            listed = [{"id": v} for v in voices]
            return httpx.Response(200, json={"voices": ["default"], "uploaded_voices": listed})
        if request.url.path == "/v1/audio/voices" and request.method == "POST":
            content = request.read()
            if uploads is not None:
                uploads.append(content)
            name = content.split(b'name="name"\r\n\r\n', 1)[1].split(b"\r\n", 1)[0]
            voices.append(name.decode())
            return httpx.Response(200, json={"success": True, "voice": {"id": name.decode()}})
        if request.url.path == "/v1/audio/speech":
            body = json.loads(request.content)
            if speech is not None:
                return speech(body)
            return httpx.Response(
                200, content=_pcm(1.0), headers={"X-Audio-Sample-Rate": str(_SAMPLE_RATE)}
            )
        raise AssertionError(f"unexpected path {request.url.path}")

    return httpx.MockTransport(handler)


def _recording(
    bodies: list[dict[str, object]],
    seconds: float = 1.0,
    *,
    max_model_len: object = _MAX_MODEL_LEN,
) -> httpx.MockTransport:
    lock = threading.Lock()

    def speech(body: dict[str, object]) -> httpx.Response:
        with lock:
            bodies.append(body)
        return httpx.Response(200, content=_pcm(seconds))

    return _transport(speech, max_model_len=max_model_len)


def test_synthesize_sends_one_cloned_request_per_text_in_order(tmp_path: Path) -> None:
    lengths = {"one": 0.5, "two two": 1.0, "three three three": 1.5}

    def speech(body: dict[str, object]) -> httpx.Response:
        return httpx.Response(200, content=_pcm(lengths[str(body["input"])]))

    with DramaEngine(_config(tmp_path), transport=_transport(speech)) as engine:
        clips = engine.synthesize(list(lengths))

    assert [round(c.seconds, 2) for c in clips] == [0.5, 1.0, 1.5]
    assert {c.sample_rate for c in clips} == {_SAMPLE_RATE}


def test_request_body_carries_voice_seed_and_per_text_cap(tmp_path: Path) -> None:
    bodies: list[dict[str, object]] = []
    voices: list[str] = []
    lock = threading.Lock()

    def speech(body: dict[str, object]) -> httpx.Response:
        with lock:
            bodies.append(body)
        return httpx.Response(200, content=_pcm(1.0))

    transport = _transport(speech, voices=voices)
    with DramaEngine(_config(tmp_path, seed=7), transport=transport) as engine:
        engine.synthesize(["short", "a much longer sentence than the other"])

    by_text = {b["input"]: b for b in bodies}
    for text, body in by_text.items():
        assert body["model"] == _MODEL
        assert body["voice"] == voices[0]
        assert "ref_audio" not in body
        assert body["response_format"] == "pcm"
        assert body["seed"] == 7
        expected = guard.max_new_tokens([str(text)], guard.DEFAULT_RUNAWAY_POLICY, _FRAME_RATE)
        assert body["max_new_tokens"] == expected
    assert (
        by_text["short"]["max_new_tokens"]
        != by_text["a much longer sentence than the other"]["max_new_tokens"]
    )


def test_reference_voice_is_uploaded_once_with_its_transcript(tmp_path: Path) -> None:
    voices: list[str] = []
    uploads: list[bytes] = []
    transport = _transport(voices=voices, uploads=uploads)
    with DramaEngine(_config(tmp_path), transport=transport):
        pass
    with DramaEngine(_config(tmp_path), transport=transport):
        pass

    assert len(uploads) == 1
    assert (tmp_path / "voice.wav").read_bytes() in uploads[0]
    assert b"Reference words." in uploads[0]
    assert b'filename="voice.wav"' in uploads[0]
    assert voices[0].startswith("e2m-")


def test_changed_transcript_uploads_a_new_voice(tmp_path: Path) -> None:
    voices: list[str] = []
    transport = _transport(voices=voices)
    with DramaEngine(_config(tmp_path), transport=transport):
        pass
    with DramaEngine(_config(tmp_path, reference_text="Other words."), transport=transport):
        pass

    assert len(set(voices)) == 2


def test_failed_voice_upload_raises(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200)
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": _MODEL, "max_model_len": 8192}]})
        if request.method == "GET":
            return httpx.Response(200, json={"voices": [], "uploaded_voices": []})
        return httpx.Response(400, json={"error": {"message": "audio too long"}})

    with pytest.raises(TTSError, match=r"voice\.wav failed with status 400.*too long"):
        DramaEngine(_config(tmp_path), transport=httpx.MockTransport(handler))


def test_batch_texts_are_posted_concurrently(tmp_path: Path) -> None:
    texts = [f"sentence {i}" for i in range(4)]
    # Every request waits until all four are in flight; a sequential engine
    # would time the barrier out.
    barrier = threading.Barrier(len(texts), timeout=10)

    def speech(body: dict[str, object]) -> httpx.Response:
        barrier.wait()
        return httpx.Response(200, content=_pcm(0.5))

    with DramaEngine(_config(tmp_path), transport=_transport(speech)) as engine:
        clips = engine.synthesize(texts)

    assert len(clips) == len(texts)


def test_batching_follows_config_and_two_batches_run_at_once(tmp_path: Path) -> None:
    with DramaEngine(_config(tmp_path, batch_size=16), transport=_transport()) as engine:
        assert engine.max_batch == 16
        assert engine.max_concurrency == 2


def test_resynthesize_reseeds_per_round_and_lifts_the_cap(tmp_path: Path) -> None:
    bodies: list[dict[str, object]] = []
    with DramaEngine(_config(tmp_path, seed=42), transport=_recording(bodies)) as engine:
        engine.resynthesize(["again"], 3, capped=True)
        engine.resynthesize(["again"], 4, capped=False)

    capped, uncapped = bodies
    assert capped["seed"] == 45
    assert uncapped["seed"] == 46
    assert int(str(uncapped["max_new_tokens"])) > int(str(capped["max_new_tokens"]))


def test_retry_hooks_follow_runaway_policy(tmp_path: Path) -> None:
    policy = guard.RunawayPolicy(retries=5)
    text = "a runaway sentence"
    limit = guard.retry_limit_seconds(text, policy)
    over = AudioClip(
        samples=np.zeros(int((limit + 1.0) * _SAMPLE_RATE), dtype=np.float32),
        sample_rate=_SAMPLE_RATE,
    )
    with DramaEngine(_config(tmp_path), transport=_transport(), runaway=policy) as engine:
        assert engine.retries == 5
        assert math.isinf(engine.clip_miss(text, over, capped=True))
        assert engine.clip_miss(text, over, capped=False) == pytest.approx(1.0, abs=1e-3)


def test_wrong_model_on_port_raises_naming_both(tmp_path: Path) -> None:
    with pytest.raises(TTSError, match=r"Qwen/Other.*fishaudio/drama-3"):
        DramaEngine(_config(tmp_path), transport=_transport(models=["Qwen/Other"]))


def test_models_route_missing_raises(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200)
        return httpx.Response(404, json={"detail": "Not Found"})

    with pytest.raises(TTSError, match="no models"):
        DramaEngine(_config(tmp_path), transport=httpx.MockTransport(handler))


def test_missing_reference_audio_raises_naming_the_path(tmp_path: Path) -> None:
    missing = tmp_path / "nope.wav"
    with pytest.raises(TTSError, match=r"nope\.wav"):
        DramaEngine(_config(tmp_path, reference_audio=missing), transport=_transport())


def test_error_status_raises_with_body_excerpt(tmp_path: Path) -> None:
    def speech(body: dict[str, object]) -> httpx.Response:
        return httpx.Response(400, json={"error": {"message": "bad seed"}})

    with (
        DramaEngine(_config(tmp_path), transport=_transport(speech)) as engine,
        pytest.raises(TTSError, match=r"status 400.*bad seed"),
    ):
        engine.synthesize(["hello"])


def test_unexpected_sample_rate_raises(tmp_path: Path) -> None:
    def speech(body: dict[str, object]) -> httpx.Response:
        return httpx.Response(200, content=_pcm(0.5), headers={"X-Audio-Sample-Rate": "24000"})

    with (
        DramaEngine(_config(tmp_path), transport=_transport(speech)) as engine,
        pytest.raises(TTSError, match="24000 Hz"),
    ):
        engine.synthesize(["hello"])


def test_fingerprint_follows_voice_seed_and_model(tmp_path: Path) -> None:
    def fingerprint(**overrides: object) -> str:
        with DramaEngine(_config(tmp_path, **overrides), transport=_transport()) as engine:
            return engine.fingerprint()

    base = fingerprint()
    other_ref = tmp_path / "other.wav"
    _wav(other_ref, level=0.2)

    assert fingerprint() == base
    assert fingerprint(seed=1) != base
    assert fingerprint(reference_text="Other words.") != base
    assert fingerprint(reference_audio=other_ref) != base
    assert fingerprint(port=9000) == base


def test_fingerprint_follows_served_model_id(tmp_path: Path) -> None:
    with DramaEngine(_config(tmp_path), transport=_transport()) as engine:
        base = engine.fingerprint()
    transport = _transport(models=["fishaudio/drama-3-int4"])
    config = _config(tmp_path, model="fishaudio/drama-3-int4")
    with DramaEngine(config, transport=transport) as engine:
        assert engine.fingerprint() != base


def _prompt_rows(text: str) -> int:
    # Reference frames + transcript + text + the engine's template allowance.
    return math.ceil(_REF_SECONDS * _FRAME_RATE) + len("Reference words.") + len(text) + 256


def test_uncapped_reseed_gets_all_the_context_left(tmp_path: Path) -> None:
    bodies: list[dict[str, object]] = []
    transport = _recording(bodies, max_model_len=_SMALL_MODEL_LEN)
    with DramaEngine(_config(tmp_path), transport=transport) as engine:
        engine.resynthesize(["again"], 1, capped=False)
    assert bodies[0]["max_new_tokens"] == _SMALL_MODEL_LEN - _prompt_rows("again")


def test_uncapped_reseed_never_asks_past_the_server_cap(tmp_path: Path) -> None:
    # 8192 rows leave far more than the server accepts as max_new_tokens.
    assert _MAX_MODEL_LEN - _prompt_rows("again") > _MAX_NEW_TOKENS
    bodies: list[dict[str, object]] = []
    with DramaEngine(_config(tmp_path), transport=_recording(bodies)) as engine:
        engine.resynthesize(["again"], 1, capped=False)
    assert bodies[0]["max_new_tokens"] == _MAX_NEW_TOKENS


@pytest.mark.parametrize("max_model_len", [_SMALL_MODEL_LEN, _MAX_MODEL_LEN])
def test_max_chars_is_the_longest_text_that_fits(tmp_path: Path, max_model_len: int) -> None:
    policy = guard.DEFAULT_RUNAWAY_POLICY

    def frames(chars: int) -> int:
        return math.ceil(guard.retry_limit_seconds("x" * chars, policy) * _FRAME_RATE)

    def room(chars: int) -> int:
        # Sized for 4-byte characters: each is four prompt rows.
        return min(max_model_len - _prompt_rows("x" * 4 * chars), _MAX_NEW_TOKENS)

    transport = _transport(max_model_len=max_model_len)
    with DramaEngine(_config(tmp_path), transport=transport) as engine:
        n = engine.max_chars
    assert frames(n) <= room(n)
    assert frames(n + 1) > room(n + 1)


@pytest.mark.parametrize("char", ["x", "你", "\U0001f600"])
def test_longest_text_in_any_script_gets_room_for_its_retry_limit(
    tmp_path: Path, char: str
) -> None:
    bodies: list[dict[str, object]] = []
    with DramaEngine(_config(tmp_path), transport=_recording(bodies)) as engine:
        text = char * engine.max_chars
        engine.synthesize([text])
        engine.resynthesize([text], 1, capped=False)
    capped, uncapped = (int(str(body["max_new_tokens"])) for body in bodies)
    limit = math.ceil(guard.retry_limit_seconds(text, guard.DEFAULT_RUNAWAY_POLICY) * _FRAME_RATE)
    assert capped <= _MAX_NEW_TOKENS
    assert limit <= uncapped <= _MAX_NEW_TOKENS


def test_capped_request_never_asks_past_the_context(tmp_path: Path) -> None:
    # 100 rows past the reference: a 20-char text leaves 80, below its
    # first-pass cap, so the cap shrinks to what the context still holds.
    text = "x" * 20
    assert guard.max_new_tokens([text], guard.DEFAULT_RUNAWAY_POLICY, _FRAME_RATE) > 80
    bodies: list[dict[str, object]] = []
    lock = threading.Lock()

    def speech(body: dict[str, object]) -> httpx.Response:
        with lock:
            bodies.append(body)
        return httpx.Response(200, content=_pcm(1.0))

    transport = _transport(speech, max_model_len=_prompt_rows("") + 100)
    with DramaEngine(_config(tmp_path), transport=transport) as engine:
        engine.synthesize([text])
    assert bodies[0]["max_new_tokens"] == 80


@pytest.mark.parametrize("bad", [None, "8192", True])
def test_missing_max_model_len_raises(tmp_path: Path, bad: object) -> None:
    with pytest.raises(TTSError, match="max_model_len"):
        DramaEngine(_config(tmp_path), transport=_transport(max_model_len=bad))


def test_reference_longer_than_the_context_raises(tmp_path: Path) -> None:
    with pytest.raises(TTSError, match="shorter clip"):
        DramaEngine(_config(tmp_path), transport=_transport(max_model_len=100))


def test_undecodable_reference_raises_naming_the_path(tmp_path: Path) -> None:
    bad = tmp_path / "garbage.wav"
    bad.write_bytes(b"not audio")
    with pytest.raises(TTSError, match=r"garbage\.wav"):
        DramaEngine(_config(tmp_path, reference_audio=bad), transport=_transport())


def test_non_ascii_text_budgets_by_utf8_bytes(tmp_path: Path) -> None:
    bodies: list[dict[str, object]] = []
    text = "你好"  # two characters, six UTF-8 bytes
    transport = _recording(bodies, max_model_len=_SMALL_MODEL_LEN)
    with DramaEngine(_config(tmp_path), transport=transport) as engine:
        engine.resynthesize([text], 1, capped=False)
    assert bodies[0]["max_new_tokens"] == _SMALL_MODEL_LEN - _prompt_rows("x" * 6)


def test_context_too_small_for_any_text_raises(tmp_path: Path) -> None:
    # Room for the prompt, but not for one character's retry limit.
    with pytest.raises(TTSError, match="no room"):
        DramaEngine(_config(tmp_path), transport=_transport(max_model_len=_prompt_rows("") + 30))


def test_zero_batch_size_raises(tmp_path: Path) -> None:
    with pytest.raises(TTSError, match="batch_size"):
        DramaEngine(_config(tmp_path, batch_size=0), transport=_transport())


def test_transport_error_during_setup_raises_tts_error(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200)
        raise httpx.ConnectError("connection reset")

    with pytest.raises(TTSError, match=r"GET .*/v1/models failed: ConnectError"):
        DramaEngine(_config(tmp_path), transport=httpx.MockTransport(handler))


def test_non_integer_sample_rate_header_raises_tts_error(tmp_path: Path) -> None:
    def speech(body: dict[str, object]) -> httpx.Response:
        return httpx.Response(200, content=_pcm(0.5), headers={"X-Audio-Sample-Rate": "fast"})

    with (
        DramaEngine(_config(tmp_path), transport=_transport(speech)) as engine,
        pytest.raises(TTSError, match="fast Hz"),
    ):
        engine.synthesize(["hello"])


def test_speech_is_resent_once_after_a_reset_connection(tmp_path: Path) -> None:
    calls = [0]

    def speech(_body: dict[str, object]) -> httpx.Response:
        calls[0] += 1
        if calls[0] == 1:
            raise httpx.ReadError("[Errno 104] Connection reset by peer")
        return httpx.Response(200, content=_pcm(1.0))

    with DramaEngine(_config(tmp_path), transport=_transport(speech)) as engine:
        (clip,) = engine.synthesize(["Hello there."])
    assert round(clip.seconds, 2) == 1.0
    assert calls[0] == 2


def test_speech_reset_twice_raises(tmp_path: Path) -> None:
    def speech(_body: dict[str, object]) -> httpx.Response:
        raise httpx.ReadError("[Errno 104] Connection reset by peer")

    with (
        DramaEngine(_config(tmp_path), transport=_transport(speech)) as engine,
        pytest.raises(TTSError, match=r"POST .*/v1/audio/speech failed: ReadError"),
    ):
        engine.synthesize(["Hello there."])
