"""App configuration: optional TOML file + env var + CLI merge.

Resolution order for *which* file to read (first match wins):

1. an explicit ``--config`` path passed on the CLI
2. the ``E2M_CONFIG`` environment variable
3. ``~/.config/epub-to-m4b/config.toml`` (fine if it doesn't exist)

Only ``[engine.<name>]`` tables are understood so far - just enough to
select and configure ``--engine breeze``. Each engine keeps its own frozen
config dataclass (``BreezeConfig`` today; ``OpenAIConfig``/``ElevenLabsConfig``/
``DeepgramConfig`` land the same way in a later milestone, one new
``_build_<name>_config`` + ``AppConfig`` field each - nothing generic beyond
that is needed yet). This module's job is only to read the matching TOML
table and merge it onto the dataclass, raising ``ConfigError`` with a clear
message when required keys are missing rather than letting a bare
TypeError/KeyError reach the user as a stack trace.
"""

from __future__ import annotations

import os
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from epub_to_m4b.tts.breeze import BreezeConfig

_CONFIG_ENV_VAR = "E2M_CONFIG"
DEFAULT_CONFIG_PATH = Path("~/.config/epub-to-m4b/config.toml").expanduser()
# Not read from TOML (it's not a per-engine tuning knob, it's where *this*
# tool keeps its own state) - matches the resume cache layout documented in
# AGENTS.md/epub-to-m4b.md: ~/.cache/epub-to-m4b/clips/... . BreezeEngine
# additionally keeps its spawned server's log and reference-voice wav under
# here (cache_dir / "breeze").
DEFAULT_CACHE_DIR = Path("~/.cache/epub-to-m4b").expanduser()

# Keys copied onto BreezeConfig verbatim (after light type coercion below).
# cache_dir is deliberately excluded - see DEFAULT_CACHE_DIR above.
_BREEZE_KEYS = ("weights_dir", "command", "port", "instruction", "cfg_scale", "seed", "batch_size")
_BREEZE_REQUIRED = ("weights_dir", "command")


class ConfigError(Exception):
    """A user-facing config problem: bad path, malformed TOML, missing keys."""


@dataclass(frozen=True)
class AppConfig:
    """Resolved config for one run. One optional field per pluggable engine.

    ``None`` means "no ``[engine.<name>]`` table was configured" - engines
    with no required fields (silence/tone) don't need one; engines that do
    (breeze) raise ``ConfigError`` from their registry factory instead of a
    confusing attribute/type error deeper in the engine.
    """

    breeze: BreezeConfig | None = None


def resolve_config_path(cli_path: Path | None) -> Path:
    """CLI flag > ``E2M_CONFIG`` env var > default path, in that order."""
    if cli_path is not None:
        return cli_path.expanduser()
    env_value = os.environ.get(_CONFIG_ENV_VAR)
    if env_value:
        return Path(env_value).expanduser()
    return DEFAULT_CONFIG_PATH


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


def _build_breeze_config(table: Mapping[str, object], *, cache_dir: Path) -> BreezeConfig:
    unknown = sorted(set(table) - set(_BREEZE_KEYS))
    if unknown:
        raise ConfigError(f"[engine.breeze]: unknown key(s): {', '.join(unknown)}")
    missing = [key for key in _BREEZE_REQUIRED if key not in table]
    if missing:
        raise ConfigError(
            f"[engine.breeze] is missing required key(s): {', '.join(missing)} - "
            "breeze ships no default command, so it must be fully configured before "
            "it can be selected with --engine breeze"
        )
    kwargs: dict[str, object] = {"cache_dir": cache_dir}
    for key, raw_value in table.items():
        coerced: object = raw_value
        if key == "weights_dir":
            coerced = Path(str(raw_value)).expanduser()
        elif key == "command":
            if not isinstance(raw_value, list) or not raw_value:
                raise ConfigError("[engine.breeze].command must be a non-empty array of strings")
            coerced = [str(part) for part in raw_value]
        kwargs[key] = coerced
    return BreezeConfig(**kwargs)  # type: ignore[arg-type]


def load_config(cli_path: Path | None, *, cache_dir: Path = DEFAULT_CACHE_DIR) -> AppConfig:
    """Read the resolved config file (if any) and build per-engine configs.

    A missing env/default-resolved path just means "no config" (breeze stays
    ``None`` - fine, silence/tone don't need one). An explicit ``--config``
    path (or ``E2M_CONFIG``) that doesn't exist, malformed TOML, or a
    ``[engine.breeze]`` table missing required keys are all ``ConfigError``.
    """
    path = resolve_config_path(cli_path)
    data = _load_toml(path, explicit=_is_explicit(cli_path))
    breeze_table = _sub_table(data, "engine", "breeze")
    breeze = _build_breeze_config(breeze_table, cache_dir=cache_dir) if breeze_table else None
    return AppConfig(breeze=breeze)
