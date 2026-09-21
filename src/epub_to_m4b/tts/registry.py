"""Engine name -> factory lookup used by the CLI's ``--engine`` flag.

Every factory takes the same ``AppConfig`` and returns a ``TTSEngine`` - one
signature for engines that need config (breeze, openai, elevenlabs), ones
that default fully (deepgram) and ones that ignore it (silence/tone). The
hosted API factories are also the only place API keys are read from the
environment: each ``*Config`` names its variable via ``api_key_env`` and the
factory hands the resolved key to the engine, so engines themselves never
touch ``os.environ``.
"""

from __future__ import annotations

import os
from collections.abc import Callable

from epub_to_m4b.config import AppConfig, ConfigError
from epub_to_m4b.tts.base import TTSEngine
from epub_to_m4b.tts.breeze import BreezeEngine
from epub_to_m4b.tts.deepgram import DeepgramConfig, DeepgramEngine
from epub_to_m4b.tts.elevenlabs import ElevenLabsEngine
from epub_to_m4b.tts.fake import SilenceEngine, ToneEngine
from epub_to_m4b.tts.openai_compat import OpenAIEngine


def _breeze_factory(config: AppConfig) -> TTSEngine:
    if config.breeze is None:
        raise ConfigError(
            "engine 'breeze' selected but no [engine.breeze] table was found - "
            "pass --config pointing at a TOML file with a fully configured "
            "[engine.breeze] table (weights_dir, command, ...) to use it"
        )
    return BreezeEngine(config.breeze)


def _missing_table(engine_name: str, required: str) -> ConfigError:
    return ConfigError(
        f"engine '{engine_name}' selected but no [engine.{engine_name}] table was found - "
        f"pass --config pointing at a TOML file with an [engine.{engine_name}] table "
        f"({required}) to use it"
    )


def _api_key(api_key_env: str, engine_name: str) -> str:
    key = os.environ.get(api_key_env)
    if not key:
        raise ConfigError(
            f"environment variable {api_key_env} is not set (needed for engine '{engine_name}')"
        )
    return key


def _openai_factory(config: AppConfig) -> TTSEngine:
    if config.openai is None:
        raise _missing_table(OpenAIEngine.name, "base_url, model, voice")
    return OpenAIEngine(
        config.openai, api_key=_api_key(config.openai.api_key_env, OpenAIEngine.name)
    )


def _elevenlabs_factory(config: AppConfig) -> TTSEngine:
    if config.elevenlabs is None:
        raise _missing_table(ElevenLabsEngine.name, "voice_id")
    return ElevenLabsEngine(
        config.elevenlabs, api_key=_api_key(config.elevenlabs.api_key_env, ElevenLabsEngine.name)
    )


def _deepgram_factory(config: AppConfig) -> TTSEngine:
    deepgram = config.deepgram if config.deepgram is not None else DeepgramConfig()
    return DeepgramEngine(deepgram, api_key=_api_key(deepgram.api_key_env, DeepgramEngine.name))


_ENGINES: dict[str, Callable[[AppConfig], TTSEngine]] = {
    SilenceEngine.name: lambda _config: SilenceEngine(),
    ToneEngine.name: lambda _config: ToneEngine(),
    BreezeEngine.name: _breeze_factory,
    OpenAIEngine.name: _openai_factory,
    ElevenLabsEngine.name: _elevenlabs_factory,
    DeepgramEngine.name: _deepgram_factory,
}


def available_engines() -> list[str]:
    return sorted(_ENGINES)


def create_engine(name: str, config: AppConfig) -> TTSEngine:
    try:
        factory = _ENGINES[name]
    except KeyError:
        raise ValueError(f"unknown engine: {name!r}") from None
    return factory(config)
