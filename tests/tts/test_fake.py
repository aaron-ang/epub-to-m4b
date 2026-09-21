from __future__ import annotations

from epub_to_m4b.tts.fake import SilenceEngine, ToneEngine

_SHORT = "hi."
_LONG = "This is a considerably longer sentence than the very short one above it."


def test_silence_engine_default_sample_rate() -> None:
    assert SilenceEngine().sample_rate == 24000


def test_silence_engine_configurable_sample_rate() -> None:
    assert SilenceEngine(sample_rate=16000).sample_rate == 16000


def test_silence_engine_is_all_zero() -> None:
    (clip,) = SilenceEngine().synthesize([_SHORT])
    assert clip.sample_rate == 24000
    assert len(clip.samples) > 0
    assert (clip.samples == 0).all()


def test_silence_engine_duration_proportional_to_text_length() -> None:
    short_clip, long_clip = SilenceEngine().synthesize([_SHORT, _LONG])
    assert long_clip.seconds > short_clip.seconds


def test_silence_engine_fingerprint_is_stable_and_deterministic() -> None:
    a = SilenceEngine(sample_rate=16000)
    b = SilenceEngine(sample_rate=16000)
    assert a.fingerprint() == b.fingerprint() == "silence:16000"


def test_silence_engine_fingerprint_reflects_sample_rate() -> None:
    low = SilenceEngine(sample_rate=16000)
    high = SilenceEngine(sample_rate=24000)
    assert low.fingerprint() != high.fingerprint()


def test_silence_engine_as_context_manager() -> None:
    with SilenceEngine() as engine:
        assert engine.synthesize([_SHORT])


def test_tone_engine_default_sample_rate() -> None:
    assert ToneEngine().sample_rate == 24000


def test_tone_engine_produces_audible_signal() -> None:
    (clip,) = ToneEngine().synthesize([_SHORT])
    assert clip.samples.any()


def test_tone_engine_duration_proportional_to_text_length() -> None:
    short_clip, long_clip = ToneEngine().synthesize([_SHORT, _LONG])
    assert long_clip.seconds > short_clip.seconds


def test_tone_engine_fingerprint_is_deterministic() -> None:
    a = ToneEngine(frequency=440.0)
    b = ToneEngine(frequency=440.0)
    assert a.fingerprint() == b.fingerprint()


def test_tone_engine_fingerprint_reflects_frequency() -> None:
    assert ToneEngine(frequency=220.0).fingerprint() != ToneEngine(frequency=440.0).fingerprint()
