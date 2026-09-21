"""Engine name -> factory lookup used by the CLI's ``--engine`` flag.

Every factory takes the same ``AppConfig`` and returns a ``TTSEngine`` - one
signature for engines that need config (breeze) and ones that don't
(silence/tone just ignore it). Real API engines register the same way once
they land: add the class + a factory that pulls its own field off
``AppConfig`` here (or, for third-party engines, via an
``epub_to_m4b.engines`` entry point - not needed yet with a single package).
"""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path

from epub_to_m4b.config import AppConfig, ConfigError
from epub_to_m4b.tts.base import TTSEngine
from epub_to_m4b.tts.breeze import BreezeEngine
from epub_to_m4b.tts.fake import SilenceEngine, ToneEngine

# Undocumented, test-only env vars - never set by normal use of the CLI.
# They exist so a resume/kill-9 test can drive `--engine silence` as a real
# subprocess (needed for a genuine SIGKILL) while still controlling its
# pacing and observing what it was asked to synthesize, neither of which is
# otherwise reachable across a process boundary.
_SILENCE_DELAY_ENV = "E2M_SILENCE_DELAY_SECONDS"
_SILENCE_CALL_LOG_ENV = "E2M_SILENCE_CALL_LOG"


def _breeze_factory(config: AppConfig) -> TTSEngine:
    if config.breeze is None:
        raise ConfigError(
            "engine 'breeze' selected but no [engine.breeze] table was found - "
            "pass --config pointing at a TOML file with a fully configured "
            "[engine.breeze] table (weights_dir, command, ...) to use it"
        )
    return BreezeEngine(config.breeze)


def _silence_factory(_config: AppConfig) -> TTSEngine:
    delay_raw = os.environ.get(_SILENCE_DELAY_ENV)
    log_raw = os.environ.get(_SILENCE_CALL_LOG_ENV)
    return SilenceEngine(
        delay_seconds=float(delay_raw) if delay_raw else 0.0,
        call_log_path=Path(log_raw) if log_raw else None,
    )


_ENGINES: dict[str, Callable[[AppConfig], TTSEngine]] = {
    SilenceEngine.name: _silence_factory,
    ToneEngine.name: lambda _config: ToneEngine(),
    BreezeEngine.name: _breeze_factory,
}


def available_engines() -> list[str]:
    return sorted(_ENGINES)


def create_engine(name: str, config: AppConfig) -> TTSEngine:
    try:
        factory = _ENGINES[name]
    except KeyError:
        raise ValueError(f"unknown engine: {name!r}") from None
    return factory(config)
