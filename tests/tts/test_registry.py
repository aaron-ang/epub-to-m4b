from __future__ import annotations

import pytest

from epub_to_m4b.config import AppConfig, ConfigError
from epub_to_m4b.tts.fake import SilenceEngine, ToneEngine
from epub_to_m4b.tts.registry import available_engines, create_engine

_NO_CONFIG = AppConfig()


def test_available_engines_includes_fakes_and_breeze() -> None:
    assert available_engines() == ["breeze", "silence", "tone"]


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
