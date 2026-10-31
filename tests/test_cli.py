from __future__ import annotations

import importlib.metadata
import logging
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import ClassVar

import pytest

from epub_to_m4b import __version__, cli
from epub_to_m4b.book import AudioClip
from epub_to_m4b.cli import build_parser, main
from epub_to_m4b.config import AppConfig
from epub_to_m4b.tts.base import TTSEngine
from epub_to_m4b.tts.http import TTSError
from epub_to_m4b.tts.registry import create_engine
from tests.conftest import build_epub


def test_parser_builds() -> None:
    assert build_parser().prog == "epub-to-m4b"


def test_no_command_exits_1(capsys: pytest.CaptureFixture[str]) -> None:
    assert main([]) == 1


def test_chapters_table(tiny_epub: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["chapters", str(tiny_epub)]) == 0
    out = capsys.readouterr().out
    assert "Tiny Book — Ada Author (2 chapters)" in out
    assert "One: The Beginning" in out and "Two: The End" in out


def test_chapters_title_column_fits_longest_title(
    tiny_epub: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["chapters", str(tiny_epub)]) == 0
    lines = capsys.readouterr().out.splitlines()
    width = len("One: The Beginning")
    assert lines[1] == f"{'#':>3}  {'title':<{width}}  {'paras':>5}  {'chars':>7}  {'docs':>4}"
    assert lines[2].startswith(f"  1  {'One: The Beginning':<{width}}  ")


def test_dump_text_single_chapter(tiny_epub: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["dump-text", str(tiny_epub), "--chapter", "2"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("=== [2] Two: The End\n# Chapter Two\n")
    assert "Chapter One" not in out


def test_dump_text_normalized_prints_each_paragraph_normalized_in_one_call(
    tiny_epub: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[list[str]] = []

    def fake_normalize_all(texts: list[str]) -> list[str]:
        calls.append(texts)
        return [f"<{i}>" for i in range(len(texts))]

    monkeypatch.setattr("epub_to_m4b.cli.normalize_all", fake_normalize_all)
    assert main(["dump-text", str(tiny_epub), "--normalized"]) == 0
    printed = [
        line.removeprefix("# ")
        for line in capsys.readouterr().out.splitlines()
        if line and not line.startswith("===")
    ]
    assert len(calls) == 1
    assert printed == [f"<{i}>" for i in range(len(calls[0]))]


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


def test_convert_help_documents_every_flag(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["convert", "--help"])
    assert excinfo.value.code == 0
    # argparse wraps long help lines at the terminal width; compare unwrapped.
    out = " ".join(capsys.readouterr().out.split())
    assert "paths to one or more .epub files" in out
    assert "TOC nesting level that starts a new chapter (default: 1)" in out
    assert "merge into a neighbour (default: 200)" in out
    assert "TTS engine (see README for configuration)" in out
    assert "directory for the .m4b and .vtt" in out


def test_version_flag_prints_installed_package_version(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert __version__ == importlib.metadata.version("epub-to-m4b")
    with pytest.raises(SystemExit) as excinfo:
        main(["--version"])
    assert excinfo.value.code == 0
    assert capsys.readouterr().out.strip() == __version__


def test_run_logged_reports_100_percent_at_total(monkeypatch: pytest.MonkeyPatch) -> None:
    # (total * 100) // total floors to 99.0 for this total, which once
    # labelled the final line "99%" at a position equal to the total.
    total = 25423.164674380045

    def fake_run(_args: Sequence[str], progress: Callable[[float], None]) -> None:
        progress(total / 2)
        progress(total)

    monkeypatch.setattr(cli, "run_with_progress", fake_run)
    lines: list[str] = []
    cli._run_logged(["ffmpeg"], "encode", total, lines.append)
    assert lines == ["encode 50% (3:31:51 / 7:03:43)", "encode 100% (7:03:43 / 7:03:43)"]


class _CountingEngine(TTSEngine):
    """Silence engine wrapper that counts opens and closes and can fail the
    first ``fail_calls`` synthesize calls."""

    name: ClassVar[str] = "counting"

    def __init__(self, inner: TTSEngine, *, fail_calls: int = 0) -> None:
        self.inner = inner
        self.sample_rate = inner.sample_rate
        self.fail_calls = fail_calls
        self.closed = 0

    def synthesize(self, texts: Sequence[str]) -> list[AudioClip]:
        if self.fail_calls:
            self.fail_calls -= 1
            raise TTSError("boom")
        return self.inner.synthesize(texts)

    def fingerprint(self) -> str:
        return self.inner.fingerprint()

    def close(self) -> None:
        self.closed += 1


@pytest.fixture
def counting(monkeypatch: pytest.MonkeyPatch) -> Callable[..., list[_CountingEngine]]:
    """Route ``create_engine`` through ``_CountingEngine``; returns the engines made."""

    def install(*, fail_calls: int = 0) -> list[_CountingEngine]:
        made: list[_CountingEngine] = []

        def factory(name: str, config: AppConfig) -> TTSEngine:
            engine = _CountingEngine(create_engine(name, config), fail_calls=fail_calls)
            made.append(engine)
            return engine

        monkeypatch.setattr("epub_to_m4b.cli.create_engine", factory)
        return made

    return install


def _convert(*paths: Path, out_dir: Path) -> int:
    return main(["convert", *map(str, paths), "--engine", "silence", "-o", str(out_dir)])


def test_convert_batch_writes_every_book_with_one_engine(
    tmp_path: Path, counting: Callable[..., list[_CountingEngine]]
) -> None:
    made = counting()
    first = build_epub(tmp_path / "a.epub", title="First Book")
    second = build_epub(tmp_path / "b.epub", title="Second Book")
    assert _convert(first, second, out_dir=tmp_path / "out") == 0
    assert (tmp_path / "out" / "first-book.m4b").is_file()
    assert (tmp_path / "out" / "second-book.m4b").is_file()
    assert len(made) == 1
    assert made[0].closed == 1


def test_convert_batch_skips_unreadable_book_and_exits_1(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    counting: Callable[..., list[_CountingEngine]],
) -> None:
    counting()
    missing = tmp_path / "missing.epub"
    good = build_epub(tmp_path / "good.epub", title="Good Book")
    assert _convert(missing, good, out_dir=tmp_path / "out") == 1
    assert (tmp_path / "out" / "good-book.m4b").is_file()
    err = capsys.readouterr().err
    assert f"cannot read {missing}" in err
    assert f"failed 1 of 2 books:\n  {missing}\n" in err


def test_convert_batch_continues_after_a_book_fails_to_synthesize(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    counting: Callable[..., list[_CountingEngine]],
) -> None:
    counting(fail_calls=1)
    first = build_epub(tmp_path / "a.epub", title="First Book")
    second = build_epub(tmp_path / "b.epub", title="Second Book")
    assert _convert(first, second, out_dir=tmp_path / "out") == 1
    assert not (tmp_path / "out" / "first-book.m4b").exists()
    assert (tmp_path / "out" / "second-book.m4b").is_file()
    err = capsys.readouterr().err
    assert "error: boom" in err
    assert f"failed 1 of 2 books:\n  {first}\n" in err


def test_convert_batch_skips_a_book_whose_output_name_is_taken(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    counting: Callable[..., list[_CountingEngine]],
) -> None:
    counting()
    first = build_epub(tmp_path / "a.epub")
    # A different book whose title slugifies to the same name.
    twin = build_epub(tmp_path / "b.epub", title="Tiny Book!")
    assert _convert(first, twin, out_dir=tmp_path / "out") == 1
    assert (tmp_path / "out" / "tiny-book.m4b").is_file()
    err = capsys.readouterr().err
    assert f"error: {twin}: tiny-book.m4b is also the output of {first}" in err


def test_convert_with_no_readable_book_never_starts_an_engine(
    tmp_path: Path, counting: Callable[..., list[_CountingEngine]]
) -> None:
    made = counting()
    assert _convert(tmp_path / "a.epub", tmp_path / "b.epub", out_dir=tmp_path / "out") == 1
    assert made == []


def test_convert_batch_converts_a_book_named_twice_once(
    tmp_path: Path, counting: Callable[..., list[_CountingEngine]]
) -> None:
    counting()
    book = build_epub(tmp_path / "a.epub")
    assert _convert(book, book, out_dir=tmp_path / "out") == 0
    assert (tmp_path / "out" / "tiny-book.m4b").is_file()


def test_convert_batch_skips_a_book_that_fails_with_an_os_error(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    counting: Callable[..., list[_CountingEngine]],
) -> None:
    counting()
    real_write_vtt = cli.write_vtt

    def write_vtt(cues: list[tuple[str, float, float]], path: Path) -> None:
        if path.stem == "first-book":
            raise OSError("No space left on device")
        real_write_vtt(cues, path)

    monkeypatch.setattr("epub_to_m4b.cli.write_vtt", write_vtt)
    first = build_epub(tmp_path / "a.epub", title="First Book")
    second = build_epub(tmp_path / "b.epub", title="Second Book")
    assert _convert(first, second, out_dir=tmp_path / "out") == 1
    assert (tmp_path / "out" / "second-book.vtt").is_file()
    err = capsys.readouterr().err
    assert "error: No space left on device" in err
    assert f"failed 1 of 2 books:\n  {first}\n" in err


def test_convert_batch_lists_every_book_when_setup_fails(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("shutil.which", lambda _name: None)
    first = build_epub(tmp_path / "a.epub", title="First Book")
    second = build_epub(tmp_path / "b.epub", title="Second Book")
    assert _convert(first, second, out_dir=tmp_path / "out") == 1
    err = capsys.readouterr().err
    assert "ffmpeg" in err
    assert f"failed 2 of 2 books:\n  {first}\n  {second}\n" in err
