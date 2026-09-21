from __future__ import annotations

import pytest

from epub_to_m4b.config import AppConfig, ConfigError
from epub_to_m4b.tts.deepgram import DeepgramConfig, DeepgramEngine
from epub_to_m4b.tts.elevenlabs import ElevenLabsConfig, ElevenLabsEngine
from epub_to_m4b.tts.fake import SilenceEngine, ToneEngine
from epub_to_m4b.tts.openai_compat import OpenAIConfig, OpenAIEngine
from epub_to_m4b.tts.registry import available_engines, create_engine

_NO_CONFIG = AppConfig()


def test_available_engines_lists_every_registered_engine() -> None:
    assert available_engines() == ["breeze", "deepgram", "elevenlabs", "openai", "silence", "tone"]


def test_create_engine_silence() -> None:
    assert isinstance(create_engine("silence", _NO_CONFIG), SilenceEngine)


def test_create_engine_tone() -> None:
    assert isinstance(create_engine("tone", _NO_CONFIG), ToneEngine)


def test_create_engine_unknown_raises() -> None:
    with pytest.raises(ValueError, match="unknown engine"):
        create_engine("nope", _NO_CONFIG)


def test_create_engine_breeze_without_config_raises_clear_error() -> None:
    with pytest.raises(ConfigError, match=r"engine\.breeze"):
        create_engine("breeze", _NO_CONFIG)


_OPENAI = OpenAIConfig(base_url="http://localhost:1", model="m", voice="v")


def test_create_engine_openai_without_table_raises_clear_error() -> None:
    with pytest.raises(ConfigError, match=r"\[engine\.openai\]"):
        create_engine("openai", _NO_CONFIG)


def test_create_engine_elevenlabs_without_table_raises_clear_error() -> None:
    with pytest.raises(ConfigError, match=r"\[engine\.elevenlabs\]"):
        create_engine("elevenlabs", _NO_CONFIG)


@pytest.mark.parametrize(
    ("name", "config", "env_var"),
    [
        ("openai", AppConfig(openai=_OPENAI), "OPENAI_API_KEY"),
        ("elevenlabs", AppConfig(elevenlabs=ElevenLabsConfig(voice_id="v")), "ELEVENLABS_API_KEY"),
        ("deepgram", AppConfig(deepgram=DeepgramConfig()), "DEEPGRAM_API_KEY"),
    ],
)
def test_create_engine_missing_api_key_env_names_the_variable(
    monkeypatch: pytest.MonkeyPatch, name: str, config: AppConfig, env_var: str
) -> None:
    monkeypatch.delenv(env_var, raising=False)
    with pytest.raises(ConfigError, match=env_var):
        create_engine(name, config)


def test_create_engine_empty_api_key_env_is_treated_as_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "")
    with pytest.raises(ConfigError, match="OPENAI_API_KEY"):
        create_engine("openai", AppConfig(openai=_OPENAI))


def test_create_engine_honours_custom_api_key_env(monkeypatch: pytest.MonkeyPatch) -> None:
    config = OpenAIConfig(base_url="http://localhost:1", model="m", voice="v", api_key_env="MY_KEY")
    monkeypatch.setenv("MY_KEY", "secret")
    with create_engine("openai", AppConfig(openai=config)) as engine:
        assert isinstance(engine, OpenAIEngine)
        assert engine.api_key == "secret"


def test_create_engine_elevenlabs_with_table_and_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ELEVENLABS_API_KEY", "secret")
    config = AppConfig(elevenlabs=ElevenLabsConfig(voice_id="v"))
    with create_engine("elevenlabs", config) as engine:
        assert isinstance(engine, ElevenLabsEngine)


def test_create_engine_deepgram_without_table_uses_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Construction opens no connection, so a default-configured engine can be
    # built against the real hostname without ever talking to it.
    monkeypatch.setenv("DEEPGRAM_API_KEY", "secret")
    with create_engine("deepgram", _NO_CONFIG) as engine:
        assert isinstance(engine, DeepgramEngine)
        assert engine.config == DeepgramConfig()
        assert engine.api_key == "secret"
