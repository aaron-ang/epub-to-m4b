"""Text normalization and sentence-splitting pipeline.

``TEXT_PIPELINE_VERSION`` is part of the resume cache key computed in a later
milestone (``synth/cache.py``): ``sha256(text_pipeline_version, text)``. It is
a hash of this pipeline's own source, so any edit to ``text/normalize.py``,
``text/split.py``, or ``text/lang/*`` automatically invalidates stale cached
clips from before the change, instead of relying on someone remembering to
bump a manual constant.
"""

import hashlib
from pathlib import Path

_PIPELINE_SOURCES = (
    "normalize.py",
    "split.py",
    "lang/__init__.py",
    "lang/english.py",
    "lang/tables_en.py",
)


def _pipeline_version() -> str:
    root = Path(__file__).parent
    digest = hashlib.sha256()
    for name in _PIPELINE_SOURCES:
        digest.update((root / name).read_bytes())
    return digest.hexdigest()[:16]


TEXT_PIPELINE_VERSION = _pipeline_version()
