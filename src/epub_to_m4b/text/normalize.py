"""Language-dispatching entry point for text normalization."""

from __future__ import annotations

from epub_to_m4b.text.lang import LANGUAGES


def normalize(text: str, lang: str = "en") -> str:
    try:
        normalizer = LANGUAGES[lang]
    except KeyError:
        raise ValueError(f"unsupported language: {lang!r}") from None
    return normalizer(text)
