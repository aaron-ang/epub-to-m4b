from __future__ import annotations

import logging
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import ClassVar

import pytest

from epub_to_m4b.book import AudioClip
from epub_to_m4b.cli import build_parser, main
from epub_to_m4b.tts.base import TTSEngine
from epub_to_m4b.tts.http import TTSError


def test_parser_builds() -> None:
    assert build_parser().prog == "epub-to-m4b"


def test_no_command_exits_1(capsys: pytest.CaptureFixture[str]) -> None:
    assert main([]) == 1


def test_chapters_table(tiny_epub: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["chapters", str(tiny_epub)]) == 0
    out = capsys.readouterr().out
    assert "Tiny Book — Ada Author (2 chapters)" in out
    assert "One: The Beginning" in out and "Two: The End" in out


def test_dump_text_single_chapter(tiny_epub: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["dump-text", str(tiny_epub), "--chapter", "2"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("=== [2] Two: The End\n# Chapter Two\n")
    assert "Chapter One" not in out


def test_dump_text_bad_chapter(tiny_epub: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["dump-text", str(tiny_epub), "--chapter", "9"]) == 1
    assert "out of range" in capsys.readouterr().err


def test_missing_file_exits_1(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["chapters", str(tmp_path / "nope.epub")]) == 1
    assert "cannot read" in capsys.readouterr().err


def test_convert_parser_accepts_breeze_engine_and_config_flag() -> None:
    parser = build_parser()
    args = parser.parse_args(
        ["convert", "book.epub", "--engine", "breeze", "-o", "out", "--config", "c.toml"]
    )
    assert args.engine == "breeze"
    assert args.config == Path("c.toml")


def test_convert_breeze_without_config_table_errors_cleanly(
    tiny_epub: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text("", encoding="utf-8")
    out_dir = tmp_path / "out"
    code = main(
        [
            "convert",
            str(tiny_epub),
            "--engine",
            "breeze",
            "-o",
            str(out_dir),
            "--config",
            str(config_path),
        ]
    )
    assert code == 1
    err = capsys.readouterr().err
    assert err.startswith("error:")
    assert "breeze" in err
    assert not (out_dir / "tiny-book.m4b").exists()


def test_convert_breeze_missing_config_file_errors_cleanly(
    tiny_epub: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out_dir = tmp_path / "out"
    missing_config = tmp_path / "missing.toml"
    code = main(
        [
            "convert",
            str(tiny_epub),
            "--engine",
            "breeze",
            "-o",
            str(out_dir),
            "--config",
            str(missing_config),
        ]
    )
    assert code == 1
    err = capsys.readouterr().err
    assert err.startswith("error:")
    assert "not found" in err


def test_convert_missing_ffmpeg_errors_cleanly(
    tiny_epub: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr("shutil.which", lambda _name: None)
    out_dir = tmp_path / "out"
    code = main(["convert", str(tiny_epub), "--engine", "silence", "-o", str(out_dir)])
    assert code == 1
    err = capsys.readouterr().err
    assert err.startswith("error:")
    assert "ffmpeg" in err
    assert "Traceback" not in err
    assert not out_dir.exists()


@pytest.fixture
def clean_package_logger() -> Iterator[logging.Logger]:
    """Package logger as a fresh process would see it: no handlers, NOTSET level.

    ``main()`` attaches a stderr handler once per process; without this reset an
    earlier test's call would have bound it to that test's captured stream.
    """
    package_logger = logging.getLogger("epub_to_m4b")
    saved = (list(package_logger.handlers), package_logger.level)
    package_logger.handlers.clear()
    package_logger.setLevel(logging.NOTSET)
    yield package_logger
    package_logger.handlers[:] = saved[0]
    package_logger.setLevel(saved[1])


def test_main_attaches_one_info_handler_to_package_logger(
    tiny_epub: Path, capsys: pytest.CaptureFixture[str], clean_package_logger: logging.Logger
) -> None:
    assert main(["chapters", str(tiny_epub)]) == 0
    assert clean_package_logger.level == logging.INFO
    assert len(clean_package_logger.handlers) == 1

    # a second main() in the same process must not stack a second handler
    assert main(["chapters", str(tiny_epub)]) == 0
    assert len(clean_package_logger.handlers) == 1

    capsys.readouterr()
    logging.getLogger("epub_to_m4b.tts.breeze").info("Breeze %s", "runaway clip recovered")
    assert capsys.readouterr().err == "Breeze runaway clip recovered\n"


def test_main_leaves_third_party_loggers_quiet(
    tiny_epub: Path, capsys: pytest.CaptureFixture[str], clean_package_logger: logging.Logger
) -> None:
    assert main(["chapters", str(tiny_epub)]) == 0
    capsys.readouterr()
    logging.getLogger("httpx").info("GET /health")
    assert capsys.readouterr().err == ""


class _ExplodingEngine(TTSEngine):
    name: ClassVar[str] = "exploding"
    sample_rate: int = 24000

    def synthesize(self, texts: Sequence[str]) -> list[AudioClip]:
        raise TTSError("boom")

    def fingerprint(self) -> str:
        return "exploding"


def test_convert_tts_error_after_retries_errors_cleanly(
    tiny_epub: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr("epub_to_m4b.cli.require_ffmpeg", lambda: None)
    monkeypatch.setattr("epub_to_m4b.cli.create_engine", lambda _name, _config: _ExplodingEngine())
    out_dir = tmp_path / "out"
    code = main(["convert", str(tiny_epub), "--engine", "silence", "-o", str(out_dir)])
    assert code == 1
    err = capsys.readouterr().err
    assert "error: boom" in err
    assert "Traceback" not in err


def test_convert_keyboard_interrupt_propagates(
    tiny_epub: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class _InterruptingEngine(_ExplodingEngine):
        def synthesize(self, texts: Sequence[str]) -> list[AudioClip]:
            raise KeyboardInterrupt

    monkeypatch.setattr("epub_to_m4b.cli.require_ffmpeg", lambda: None)
    monkeypatch.setattr(
        "epub_to_m4b.cli.create_engine", lambda _name, _config: _InterruptingEngine()
    )
    with pytest.raises(KeyboardInterrupt):
        main(["convert", str(tiny_epub), "--engine", "silence", "-o", str(tmp_path / "out")])
