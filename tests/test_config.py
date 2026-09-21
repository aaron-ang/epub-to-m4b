from __future__ import annotations

from pathlib import Path

import pytest

from epub_to_m4b.config import (
    DEFAULT_CONFIG_PATH,
    AppConfig,
    ConfigError,
    load_config,
    resolve_config_path,
)


def _write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def test_load_config_missing_default_path_yields_no_breeze(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Not explicitly requested (no --config, no E2M_CONFIG) - a missing file
    # at the default location just means "no config", not an error.
    monkeypatch.delenv("E2M_CONFIG", raising=False)
    monkeypatch.setattr("epub_to_m4b.config.DEFAULT_CONFIG_PATH", tmp_path / "nope.toml")
    assert load_config(None) == AppConfig(breeze=None)


def test_load_config_explicit_path_via_env_var_missing_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("E2M_CONFIG", str(tmp_path / "nope.toml"))
    with pytest.raises(ConfigError, match="not found"):
        load_config(None)


def test_load_config_no_engine_table_yields_no_breeze(tmp_path: Path) -> None:
    path = _write(tmp_path / "config.toml", "[unrelated]\nkey = 1\n")
    assert load_config(path) == AppConfig(breeze=None)


def test_load_config_merges_breeze_table_over_defaults(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "config.toml",
        f"""
        [engine.breeze]
        weights_dir = "{tmp_path / "weights"}"
        command = ["uv", "run", "breeze-infer-api"]
        port = 9999
        """,
    )
    config = load_config(path, cache_dir=tmp_path / "cache")
    assert config.breeze is not None
    assert config.breeze.weights_dir == tmp_path / "weights"
    assert config.breeze.command == ["uv", "run", "breeze-infer-api"]
    assert config.breeze.port == 9999
    # Not present in TOML - falls back to BreezeConfig's own field defaults.
    assert config.breeze.cfg_scale == 4.0
    assert config.breeze.seed == 42
    assert config.breeze.batch_size == 64
    assert config.breeze.cache_dir == tmp_path / "cache"


def test_load_config_breeze_table_missing_command_raises(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "config.toml",
        f"""
        [engine.breeze]
        weights_dir = "{tmp_path / "weights"}"
        """,
    )
    with pytest.raises(ConfigError, match="command"):
        load_config(path)


def test_load_config_breeze_table_missing_entirely_is_not_an_error(tmp_path: Path) -> None:
    path = _write(tmp_path / "config.toml", "")
    assert load_config(path).breeze is None


def test_load_config_unknown_breeze_key_raises(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "config.toml",
        """
        [engine.breeze]
        weights_dir = "/weights"
        command = ["cmd"]
        bogus = 1
        """,
    )
    with pytest.raises(ConfigError, match="bogus"):
        load_config(path)


def test_load_config_explicit_missing_path_raises(tmp_path: Path) -> None:
    missing = tmp_path / "definitely-not-here.toml"
    with pytest.raises(ConfigError, match="not found"):
        load_config(missing)


def test_load_config_invalid_toml_raises(tmp_path: Path) -> None:
    path = _write(tmp_path / "config.toml", "not valid toml [[[")
    with pytest.raises(ConfigError, match="invalid TOML"):
        load_config(path)


def test_resolve_config_path_cli_beats_env_beats_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cli_path = tmp_path / "cli.toml"
    env_path = tmp_path / "env.toml"

    monkeypatch.delenv("E2M_CONFIG", raising=False)
    assert resolve_config_path(None) == DEFAULT_CONFIG_PATH

    monkeypatch.setenv("E2M_CONFIG", str(env_path))
    assert resolve_config_path(None) == env_path

    assert resolve_config_path(cli_path) == cli_path
