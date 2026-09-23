"""App configuration: optional TOML file + env var + CLI merge.

Resolution order for *which* file to read (first match wins):

1. an explicit ``--config`` path passed on the CLI
2. the ``E2M_CONFIG`` environment variable
3. ``~/.config/epub-to-m4b/config.toml`` (fine if it doesn't exist)

Only ``[engine.<name>]`` tables are understood. Each engine keeps its own
frozen config dataclass (``BreezeConfig``, ``OpenAIConfig``,
``ElevenLabsConfig``, ``DeepgramConfig``) and ``AppConfig`` carries one
optional field per engine. ``_build_engine_config`` merges a TOML table onto
any of those dataclasses, driven by its declared fields and annotations:
unknown keys, missing required keys, and values of the wrong TOML type all
raise ``ConfigError`` with the section and key named, rather than letting a
bare TypeError/KeyError reach the user as a stack trace or a ``str`` land
on an ``int`` field. Per-engine quirks (breeze's non-empty command check
and injected cache_dir) are passed in as arguments, not special-cased
inside the builder.
"""

from __future__ import annotations

import os
import tomllib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, fields
from pathlib import Path
from typing import TYPE_CHECKING, get_args, get_origin, get_type_hints

from epub_to_m4b.errors import EpubToM4bError
from epub_to_m4b.tts.breeze import BreezeConfig
from epub_to_m4b.tts.deepgram import DeepgramConfig
from epub_to_m4b.tts.elevenlabs import ElevenLabsConfig
from epub_to_m4b.tts.openai_compat import OpenAIConfig

if TYPE_CHECKING:
    from _typeshed import DataclassInstance

_CONFIG_ENV_VAR = "E2M_CONFIG"
_CACHE_DIR_ENV_VAR = "E2M_CACHE_DIR"
DEFAULT_CONFIG_PATH = Path("~/.config/epub-to-m4b/config.toml").expanduser()
# Not read from TOML (it's not a per-engine tuning knob, it's where *this*
# tool keeps its own state) - matches the resume cache layout documented in
# AGENTS.md/epub-to-m4b.md: ~/.cache/epub-to-m4b/clips/... . BreezeEngine
# additionally keeps its spawned server's log and reference-voice wav under
# here (cache_dir / "breeze").
DEFAULT_CACHE_DIR = Path("~/.cache/epub-to-m4b").expanduser()

_BREEZE_REQUIRED = ("command",)
_OPENAI_REQUIRED = ("base_url", "model", "voice")
_ELEVENLABS_REQUIRED = ("voice_id",)


class ConfigError(EpubToM4bError):
    """A user-facing config problem: bad path, malformed TOML, missing keys."""


@dataclass(frozen=True)
class AppConfig:
    """Resolved config for one run. One optional field per pluggable engine.

    ``None`` means "no ``[engine.<name>]`` table was configured" - engines
    with no required fields (silence/tone/deepgram) don't need one; engines
    that do (breeze/openai/elevenlabs) raise ``ConfigError`` from their
    registry factory instead of a confusing attribute/type error deeper in
    the engine.
    """

    breeze: BreezeConfig | None = None
    openai: OpenAIConfig | None = None
    elevenlabs: ElevenLabsConfig | None = None
    deepgram: DeepgramConfig | None = None


def resolve_config_path(cli_path: Path | None) -> Path:
    """CLI flag > ``E2M_CONFIG`` env var > default path, in that order."""
    if cli_path is not None:
        return cli_path.expanduser()
    env_value = os.environ.get(_CONFIG_ENV_VAR)
    if env_value:
        return Path(env_value).expanduser()
    return DEFAULT_CONFIG_PATH


def resolve_cache_dir() -> Path:
    """``E2M_CACHE_DIR`` env var override of ``DEFAULT_CACHE_DIR`` - lets the
    resume clip cache live somewhere other than ``~/.cache/epub-to-m4b``
    without a config file, same override precedent as ``E2M_CONFIG``. Handy
    for tests that must not touch a real user cache, and for anyone who
    wants the cache on a different disk.
    """
    value = os.environ.get(_CACHE_DIR_ENV_VAR)
    return Path(value).expanduser() if value else DEFAULT_CACHE_DIR


def _is_explicit(cli_path: Path | None) -> bool:
    return cli_path is not None or bool(os.environ.get(_CONFIG_ENV_VAR))


def _load_toml(path: Path, *, explicit: bool) -> dict[str, object]:
    if not path.exists():
        if explicit:
            raise ConfigError(f"config file not found: {path}")
        return {}
    try:
        with path.open("rb") as toml_file:
            return tomllib.load(toml_file)
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"invalid TOML in {path}: {exc}") from exc


def _sub_table(data: Mapping[str, object], *keys: str) -> Mapping[str, object] | None:
    node: object = data
    for key in keys:
        if not isinstance(node, Mapping) or key not in node:
            return None
        node = node[key]
    return node if isinstance(node, Mapping) else None


def _coerce_command(raw_value: object) -> object:
    if not isinstance(raw_value, list) or not raw_value:
        raise ConfigError("[engine.breeze].command must be a non-empty array of strings")
    return raw_value


def _check_type(section: str, key: str, value: object, annotation: object) -> object:
    """Reject a TOML value whose type doesn't fit the dataclass field.

    ``bool`` is a subclass of ``int`` in Python but a distinct type in TOML,
    so ``true`` is refused for ``int``/``float`` fields; an ``int`` for a
    ``float`` field is converted to ``float`` because TOML has no way to
    write ``4`` as a float without ``4.0``. ``Path`` fields take a TOML
    string and get ``~`` expanded here.
    Any other annotation is a programming error in the config dataclass, not
    a user error, so it raises ``TypeError`` to force a deliberate extension.
    """
    origin = get_origin(annotation)
    if annotation in (str, bool):
        ok = type(value) is annotation
        expected = annotation.__name__
    elif annotation is int:
        ok = type(value) is int
        expected = "int"
    elif annotation is float:
        ok = type(value) in (int, float)
        expected = "float"
        if ok and isinstance(value, int | float):
            value = float(value)
    elif annotation is Path:
        ok = isinstance(value, str)
        expected = "str"
        if ok:
            value = Path(str(value)).expanduser()
    elif origin in (Sequence, list, tuple) and get_args(annotation)[:1] == (str,):
        ok = isinstance(value, list) and all(type(item) is str for item in value)
        expected = "array of str"
    else:
        raise TypeError(
            f"{section}.{key}: unsupported config field annotation {annotation!r}; "
            "extend _check_type"
        )
    if not ok:
        raise ConfigError(
            f"[engine.{section}].{key}: expected {expected}, got {type(value).__name__}"
        )
    return value


def _build_engine_config[C: DataclassInstance](
    table: Mapping[str, object],
    config_cls: type[C],
    *,
    section: str,
    required: tuple[str, ...],
    coerce: Mapping[str, Callable[[object], object]] = {},
    extra: Mapping[str, object] = {},
    missing_hint: str = "",
) -> C:
    """Merge one ``[engine.<section>]`` table onto ``config_cls``.

    The dataclass's own fields define the accepted keys, minus anything
    supplied through ``extra`` (values the tool injects itself rather than
    reads from TOML). ``coerce`` maps a key to a function applied to its raw
    TOML value before the per-field type check; every value then has to
    match the field's annotation (see ``_check_type``).
    """
    accepted = {f.name for f in fields(config_cls)} - set(extra)
    unknown = sorted(set(table) - accepted)
    if unknown:
        raise ConfigError(f"[engine.{section}]: unknown key(s): {', '.join(unknown)}")
    missing = [key for key in required if key not in table]
    if missing:
        raise ConfigError(
            f"[engine.{section}] is missing required key(s): {', '.join(missing)}{missing_hint}"
        )
    hints = get_type_hints(config_cls)
    kwargs: dict[str, object] = dict(extra)
    for key, raw_value in table.items():
        value = coerce[key](raw_value) if key in coerce else raw_value
        kwargs[key] = _check_type(section, key, value, hints[key])
    return config_cls(**kwargs)


def _build_breeze_config(table: Mapping[str, object], *, cache_dir: Path) -> BreezeConfig:
    return _build_engine_config(
        table,
        BreezeConfig,
        section="breeze",
        required=_BREEZE_REQUIRED,
        coerce={"command": _coerce_command},
        extra={"cache_dir": cache_dir},
        missing_hint=(
            " - breeze ships no default command, so it must be fully configured before "
            "it can be selected with --engine breeze"
        ),
    )


def load_config(cli_path: Path | None, *, cache_dir: Path = DEFAULT_CACHE_DIR) -> AppConfig:
    """Read the resolved config file (if any) and build per-engine configs.

    A missing env/default-resolved path just means "no config" (every engine
    field stays ``None`` - fine, silence/tone don't need one). An explicit
    ``--config`` path (or ``E2M_CONFIG``) that doesn't exist, malformed TOML,
    or an engine table with unknown or missing required keys are all
    ``ConfigError``. A present-but-empty table still yields that engine's
    all-defaults config.
    """
    path = resolve_config_path(cli_path)
    data = _load_toml(path, explicit=_is_explicit(cli_path))
    breeze_table = _sub_table(data, "engine", "breeze")
    openai_table = _sub_table(data, "engine", "openai")
    elevenlabs_table = _sub_table(data, "engine", "elevenlabs")
    deepgram_table = _sub_table(data, "engine", "deepgram")
    return AppConfig(
        breeze=(
            _build_breeze_config(breeze_table, cache_dir=cache_dir)
            if breeze_table is not None
            else None
        ),
        openai=(
            _build_engine_config(
                openai_table, OpenAIConfig, section="openai", required=_OPENAI_REQUIRED
            )
            if openai_table is not None
            else None
        ),
        elevenlabs=(
            _build_engine_config(
                elevenlabs_table,
                ElevenLabsConfig,
                section="elevenlabs",
                required=_ELEVENLABS_REQUIRED,
            )
            if elevenlabs_table is not None
            else None
        ),
        deepgram=(
            _build_engine_config(deepgram_table, DeepgramConfig, section="deepgram", required=())
            if deepgram_table is not None
            else None
        ),
    )
