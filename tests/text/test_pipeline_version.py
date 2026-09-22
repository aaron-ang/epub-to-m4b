"""``TEXT_PIPELINE_VERSION`` tracks code and data, not prose."""

from __future__ import annotations

import re
from pathlib import Path

import epub_to_m4b.text as text_pkg
from epub_to_m4b.text import TEXT_PIPELINE_VERSION, pipeline_version

BASE = b'''
"""Module docstring."""

TABLE = {"one": "1", "two": "2"}


class Thing:
    """Class docstring."""

    limit = 10

    def run(self, value):
        """Function docstring."""
        # a comment
        return value + self.limit


async def fetch(x):
    """Async docstring."""
    return x
'''


def _v(*sources: bytes) -> str:
    return pipeline_version(sources)


def test_deterministic_and_16_lowercase_hex() -> None:
    first, second = _v(BASE), _v(BASE)
    assert first == second
    assert re.fullmatch(r"[0-9a-f]{16}", first)


def test_module_docstring_does_not_change_version() -> None:
    edited = BASE.replace(b'"""Module docstring."""', b'"""A very different module docstring."""')
    assert edited != BASE
    assert _v(edited) == _v(BASE)
    removed = BASE.replace(b'"""Module docstring."""\n', b"")
    assert _v(removed) == _v(BASE)


def test_function_docstring_does_not_change_version() -> None:
    edited = BASE.replace(b'"""Function docstring."""', b'"""Reworded."""')
    assert _v(edited) == _v(BASE)
    removed = BASE.replace(b'        """Function docstring."""\n', b"")
    assert _v(removed) == _v(BASE)
    edited_async = BASE.replace(b'"""Async docstring."""', b'"""Other."""')
    assert _v(edited_async) == _v(BASE)


def test_class_docstring_does_not_change_version() -> None:
    edited = BASE.replace(b'"""Class docstring."""', b'"""Reworded class doc."""')
    assert _v(edited) == _v(BASE)
    removed = BASE.replace(b'    """Class docstring."""\n\n', b"")
    assert _v(removed) == _v(BASE)


def test_comments_do_not_change_version() -> None:
    edited = BASE.replace(b"# a comment", b"# a totally different comment")
    assert _v(edited) == _v(BASE)
    removed = BASE.replace(b"        # a comment\n", b"")
    assert _v(removed) == _v(BASE)
    added = BASE + b"\n# trailing comment\n"
    assert _v(added) == _v(BASE)


def test_whitespace_does_not_change_version() -> None:
    blank_lines = BASE.replace(b"\n\n\n", b"\n\n\n\n\n")
    assert blank_lines != BASE
    assert _v(blank_lines) == _v(BASE)
    spaces = BASE.replace(b"value + self.limit", b"value   +   self.limit")
    assert _v(spaces) == _v(BASE)


def test_non_docstring_string_literal_changes_version() -> None:
    edited = BASE.replace(b'"two": "2"', b'"two": "II"')
    assert _v(edited) != _v(BASE)


def test_variable_rename_changes_version() -> None:
    edited = BASE.replace(b"TABLE", b"LOOKUP")
    assert _v(edited) != _v(BASE)


def test_number_change_changes_version() -> None:
    edited = BASE.replace(b"limit = 10", b"limit = 11")
    assert _v(edited) != _v(BASE)


def test_added_statement_changes_version() -> None:
    edited = BASE + b"\nEXTRA = 1\n"
    assert _v(edited) != _v(BASE)


def test_source_order_matters() -> None:
    a, b = b"A = 1\n", b"B = 2\n"
    assert _v(a, b) != _v(b, a)


def test_constant_matches_pipeline_sources() -> None:
    root = Path(text_pkg.__file__).parent
    expected = pipeline_version((root / name).read_bytes() for name in text_pkg._PIPELINE_SOURCES)
    assert expected == TEXT_PIPELINE_VERSION
