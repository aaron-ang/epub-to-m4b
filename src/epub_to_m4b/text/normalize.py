"""Text normalization: run NeMo over the paragraph text.

NVIDIA NeMo text processing rewrites written forms as spoken ones (numbers,
years, dates, times, currency, measures, common abbreviations) with weighted
finite-state grammars, one set per language. Its output is used as is.
"""

from __future__ import annotations

import functools
import logging
import math
import os
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from importlib.metadata import version
from itertools import repeat
from pathlib import Path
from typing import Protocol

from epub_to_m4b.config import resolve_cache_dir

_NEMO_PACKAGE = "nemo-text-processing"
_NEMO_LOGGER = "NeMo-text-processing"
# Replaces <cache_dir>/nemo, so separate runs can share compiled grammars
# while keeping their clip caches apart.
_NEMO_CACHE_ENV_VAR = "E2M_NEMO_CACHE_DIR"


class _Normalizer(Protocol):
    def normalize(self, text: str, *, punct_post_process: bool) -> str: ...


def nemo_cache_dir(lang: str) -> Path:
    """Where NeMo keeps one language's compiled grammars:
    ``<E2M_NEMO_CACHE_DIR or cache_dir/nemo>/<package version>/<lang>``. NeMo
    loads any grammar file it finds under its cache dir by name, and two
    languages can use the same file name."""
    value = os.environ.get(_NEMO_CACHE_ENV_VAR)
    root = Path(value).expanduser() if value else resolve_cache_dir() / "nemo"
    return root / version(_NEMO_PACKAGE) / lang


@functools.cache
def nemo_normalizer(lang: str) -> _Normalizer:
    """The process-wide NeMo normalizer for ``lang``. The first call with an
    empty cache dir compiles the grammars into it; later calls load them."""
    # Imported here so commands that never normalize text skip loading NeMo.
    from nemo_text_processing.text_normalization.normalize import Normalizer  # noqa: PLC0415

    # NeMo logs every sentence and resets its logger's level on each call.
    logging.getLogger(_NEMO_LOGGER).disabled = True
    try:
        normalizer: _Normalizer = Normalizer(
            input_case="cased", lang=lang, cache_dir=str(nemo_cache_dir(lang))
        )
    except NotImplementedError:
        raise ValueError(f"unsupported language: {lang!r}") from None
    return normalizer


def _normalize_chunk(lang: str, texts: list[str]) -> list[str]:
    normalizer = nemo_normalizer(lang)
    # punct_post_process puts the input's spacing around punctuation back,
    # which NeMo's tokenizer otherwise changes.
    return [normalizer.normalize(text, punct_post_process=True) for text in texts]


def normalize_all(texts: Sequence[str], lang: str = "en") -> list[str]:
    """Normalize ``texts`` in order, one contiguous chunk per CPU core. Each
    worker loads the compiled grammars once; the workers exit on return."""
    if not texts:
        return []
    jobs = min(len(texts), os.process_cpu_count() or 1)
    if jobs == 1:
        return _normalize_chunk(lang, list(texts))
    # Compiles the grammars into the cache before the workers load them.
    nemo_normalizer(lang)
    size = math.ceil(len(texts) / jobs)
    chunks = [list(texts[i : i + size]) for i in range(0, len(texts), size)]
    with ProcessPoolExecutor(jobs, initializer=nemo_normalizer, initargs=(lang,)) as pool:
        return [
            text for chunk in pool.map(_normalize_chunk, repeat(lang), chunks) for text in chunk
        ]


def normalize(text: str, lang: str = "en") -> str:
    return normalize_all([text], lang)[0]
