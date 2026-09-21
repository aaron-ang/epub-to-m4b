from __future__ import annotations

from pathlib import Path

import pytest

from epub_to_m4b.cli import build_parser, main


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
