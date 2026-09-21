from __future__ import annotations

import pytest

from epub_to_m4b.tts.fake import SilenceEngine, ToneEngine
from epub_to_m4b.tts.registry import available_engines, create_engine


def test_available_engines_includes_fakes() -> None:
    assert available_engines() == ["silence", "tone"]


def test_create_engine_silence() -> None:
    assert isinstance(create_engine("silence"), SilenceEngine)


def test_create_engine_tone() -> None:
    assert isinstance(create_engine("tone"), ToneEngine)


def test_create_engine_unknown_raises() -> None:
    with pytest.raises(ValueError, match="unknown engine"):
        create_engine("nope")
