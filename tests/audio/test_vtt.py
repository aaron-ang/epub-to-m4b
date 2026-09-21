from __future__ import annotations

from pathlib import Path

from epub_to_m4b.audio.vtt import write_vtt


def test_write_vtt_basic_cues(tmp_path: Path) -> None:
    out = tmp_path / "out.vtt"
    write_vtt([("Hello there.", 0.0, 1.0), ("Second cue.", 1.0, 2.5)], out)
    assert out.read_text(encoding="utf-8") == (
        "WEBVTT\n"
        "\n"
        "00:00:00.000 --> 00:00:01.000\n"
        "Hello there.\n"
        "\n"
        "00:00:01.000 --> 00:00:02.500\n"
        "Second cue.\n"
    )


def test_write_vtt_crosses_minute_boundary(tmp_path: Path) -> None:
    out = tmp_path / "out.vtt"
    write_vtt([("Right at the edge.", 59.5, 60.25)], out)
    assert out.read_text(encoding="utf-8") == (
        "WEBVTT\n\n00:00:59.500 --> 00:01:00.250\nRight at the edge.\n"
    )


def test_write_vtt_millisecond_precision(tmp_path: Path) -> None:
    out = tmp_path / "out.vtt"
    write_vtt([("Precise.", 1.001, 1.999)], out)
    assert out.read_text(encoding="utf-8") == (
        "WEBVTT\n\n00:00:01.001 --> 00:00:01.999\nPrecise.\n"
    )


def test_write_vtt_empty_cues(tmp_path: Path) -> None:
    out = tmp_path / "out.vtt"
    write_vtt([], out)
    assert out.read_text(encoding="utf-8") == "WEBVTT\n"
