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
from epub_to_m4b.tts.deepgram import DeepgramConfig
from epub_to_m4b.tts.elevenlabs import ElevenLabsConfig
from epub_to_m4b.tts.openai_compat import OpenAIConfig


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


def test_load_config_parses_openai_table(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "config.toml",
        """
        [engine.openai]
        base_url = "http://localhost:8880"
        model = "kokoro"
        voice = "af_heart"
        speed = 1.1
        api_key_env = "KOKORO_KEY"
        """,
    )
    assert load_config(path).openai == OpenAIConfig(
        base_url="http://localhost:8880",
        model="kokoro",
        voice="af_heart",
        speed=1.1,
        api_key_env="KOKORO_KEY",
    )


def test_load_config_parses_elevenlabs_table_with_defaults(tmp_path: Path) -> None:
    path = _write(tmp_path / "config.toml", '[engine.elevenlabs]\nvoice_id = "abc"\n')
    assert load_config(path).elevenlabs == ElevenLabsConfig(voice_id="abc")


def test_load_config_parses_deepgram_table(tmp_path: Path) -> None:
    path = _write(tmp_path / "config.toml", '[engine.deepgram]\nmodel = "aura-2-orion-en"\n')
    assert load_config(path).deepgram == DeepgramConfig(model="aura-2-orion-en")


def test_load_config_empty_deepgram_table_still_yields_a_config(tmp_path: Path) -> None:
    path = _write(tmp_path / "config.toml", "[engine.deepgram]\n")
    assert load_config(path).deepgram == DeepgramConfig()


def test_load_config_no_api_tables_yields_none_for_each(tmp_path: Path) -> None:
    config = load_config(_write(tmp_path / "config.toml", ""))
    assert (config.openai, config.elevenlabs, config.deepgram) == (None, None, None)


@pytest.mark.parametrize(
    ("section", "body"),
    [
        ("openai", 'base_url = "x"\nmodel = "m"\nvoice = "v"\nbogus = 1'),
        ("elevenlabs", 'voice_id = "v"\nbogus = 1'),
        ("deepgram", "bogus = 1"),
    ],
)
def test_load_config_unknown_key_names_the_section(tmp_path: Path, section: str, body: str) -> None:
    path = _write(tmp_path / "config.toml", f"[engine.{section}]\n{body}\n")
    with pytest.raises(ConfigError, match=rf"\[engine\.{section}\]: unknown key\(s\): bogus"):
        load_config(path)


def test_load_config_openai_missing_required_keys_lists_them(tmp_path: Path) -> None:
    path = _write(tmp_path / "config.toml", '[engine.openai]\nmodel = "m"\n')
    with pytest.raises(ConfigError, match=r"\[engine\.openai\].*base_url, voice"):
        load_config(path)


def test_load_config_elevenlabs_missing_voice_id_raises(tmp_path: Path) -> None:
    path = _write(tmp_path / "config.toml", "[engine.elevenlabs]\n")
    with pytest.raises(ConfigError, match=r"\[engine\.elevenlabs\].*voice_id"):
        load_config(path)


def _breeze_toml(tmp_path: Path, extra_line: str) -> Path:
    return _write(
        tmp_path / "config.toml",
        f"""
        [engine.breeze]
        weights_dir = "{tmp_path / "weights"}"
        command = ["uv", "run", "breeze-infer-api"]
        {extra_line}
        """,
    )


def test_load_config_breeze_port_string_raises(tmp_path: Path) -> None:
    path = _breeze_toml(tmp_path, 'port = "abc"')
    with pytest.raises(ConfigError, match=r"\[engine\.breeze\]\.port: expected int, got str"):
        load_config(path, cache_dir=tmp_path / "cache")


def test_load_config_breeze_cfg_scale_int_accepted_for_float(tmp_path: Path) -> None:
    path = _breeze_toml(tmp_path, "cfg_scale = 4")
    config = load_config(path, cache_dir=tmp_path / "cache")
    assert config.breeze is not None
    assert config.breeze.cfg_scale == 4.0


def test_load_config_breeze_cfg_scale_bool_raises(tmp_path: Path) -> None:
    path = _breeze_toml(tmp_path, "cfg_scale = true")
    with pytest.raises(
        ConfigError, match=r"\[engine\.breeze\]\.cfg_scale: expected float, got bool"
    ):
        load_config(path, cache_dir=tmp_path / "cache")


def test_load_config_breeze_command_non_string_item_raises(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "config.toml",
        f"""
        [engine.breeze]
        weights_dir = "{tmp_path / "weights"}"
        command = ["uv", 3]
        """,
    )
    with pytest.raises(ConfigError, match=r"\[engine\.breeze\]\.command"):
        load_config(path, cache_dir=tmp_path / "cache")


def test_load_config_openai_speed_string_raises(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "config.toml",
        """
        [engine.openai]
        base_url = "http://localhost:8880"
        model = "kokoro"
        voice = "af_heart"
        speed = "fast"
        """,
    )
    with pytest.raises(ConfigError, match=r"\[engine\.openai\]\.speed: expected float, got str"):
        load_config(path)


def test_load_config_elevenlabs_voice_id_int_raises(tmp_path: Path) -> None:
    path = _write(tmp_path / "config.toml", "[engine.elevenlabs]\nvoice_id = 12\n")
    with pytest.raises(
        ConfigError, match=r"\[engine\.elevenlabs\]\.voice_id: expected str, got int"
    ):
        load_config(path)
