"""Text normalization and sentence-splitting pipeline.

``TEXT_PIPELINE_VERSION`` is part of the resume cache key computed in a later
milestone (``synth/cache.py``): ``sha256(text_pipeline_version, text)``. Bump
it whenever ``text/normalize.py`` or ``text/split.py`` changes what they
output for the same input, so a stale cache from before the change is
invalidated instead of silently reused.
"""

TEXT_PIPELINE_VERSION = "1"
