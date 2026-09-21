"""Engine name -> factory lookup used by the CLI's ``--engine`` flag.

Every factory takes the same ``AppConfig`` and returns a ``TTSEngine`` - one
signature for engines that need config (breeze) and ones that don't
(silence/tone just ignore it). Real API engines register the same way once
they land: add the class + a factory that pulls its own field off
``AppConfig`` here (or, for third-party engines, via an
``epub_to_m4b.engines`` entry point - not needed yet with a single package).
"""

from __future__ import annotations

from collections.abc import Callable

from epub_to_m4b.config import AppConfig, ConfigError
from epub_to_m4b.tts.base import TTSEngine
from epub_to_m4b.tts.breeze import BreezeEngine
from epub_to_m4b.tts.fake import SilenceEngine, ToneEngine


def _breeze_factory(config: AppConfig) -> TTSEngine:
    if config.breeze is None:
        raise ConfigError(
            "engine 'breeze' selected but no [engine.breeze] table was found - "
            "pass --config pointing at a TOML file with a fully configured "
            "[engine.breeze] table (weights_dir, command, ...) to use it"
        )
    return BreezeEngine(config.breeze)


_ENGINES: dict[str, Callable[[AppConfig], TTSEngine]] = {
    SilenceEngine.name: lambda _config: SilenceEngine(),
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
