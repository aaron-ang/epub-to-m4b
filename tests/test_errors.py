from __future__ import annotations

import pytest

from epub_to_m4b.audio.ffmpeg import FFmpegError, FFmpegNotFoundError
from epub_to_m4b.config import ConfigError
from epub_to_m4b.errors import EpubToM4bError
from epub_to_m4b.synth.orchestrator import SynthesisError
from epub_to_m4b.tts.http import TTSError
from epub_to_m4b.tts.sidecar import SidecarError


@pytest.mark.parametrize(
    "error_type",
    [ConfigError, TTSError, FFmpegNotFoundError, FFmpegError, SidecarError, SynthesisError],
)
def test_concrete_errors_share_user_facing_base(error_type: type[Exception]) -> None:
    assert isinstance(error_type("x"), EpubToM4bError)
