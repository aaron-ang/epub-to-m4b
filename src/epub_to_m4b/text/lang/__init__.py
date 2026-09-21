"""Registry of per-language normalize functions."""

from __future__ import annotations

from collections.abc import Callable

from epub_to_m4b.text.lang.english import normalize_english

LANGUAGES: dict[str, Callable[[str], str]] = {
    "en": normalize_english,
}
