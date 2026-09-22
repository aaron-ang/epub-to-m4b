"""Text normalization and sentence-splitting pipeline.

``TEXT_PIPELINE_VERSION`` is part of the resume cache key that
``synth/cache.py`` computes: ``sha256(text_pipeline_version, text)``. It is
derived from this pipeline's own source (``text/normalize.py``,
``text/split.py``, ``text/lang/*``) so a change to how text is produced
automatically invalidates stale cached clips, instead of relying on someone
remembering to bump a manual constant.

The version hashes the parsed AST with docstrings removed, not the raw
bytes. Code, data tables, string literals used as values, and identifier
names all feed the hash; comments, docstrings, blank lines, and formatting
do not, so prose-only edits leave every cached clip valid.
"""

from __future__ import annotations

import ast
import hashlib
from collections.abc import Iterable
from pathlib import Path

_PIPELINE_SOURCES = (
    "normalize.py",
    "split.py",
    "lang/__init__.py",
    "lang/english.py",
    "lang/tables_en.py",
)

_DOCSTRING_HOSTS = (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)


def _strip_docstrings(tree: ast.Module) -> ast.Module:
    for node in ast.walk(tree):
        if not isinstance(node, _DOCSTRING_HOSTS) or not node.body:
            continue
        first = node.body[0]
        if (
            isinstance(first, ast.Expr)
            and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str)
        ):
            del node.body[0]
    return tree


def pipeline_version(sources: Iterable[bytes]) -> str:
    """16-hex-char digest of the docstring-stripped AST of ``sources``, in order."""
    digest = hashlib.sha256()
    for source in sources:
        tree = _strip_docstrings(ast.parse(source))
        digest.update(ast.dump(tree, include_attributes=False).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()[:16]


def _read_pipeline_sources() -> Iterable[bytes]:
    root = Path(__file__).parent
    return ((root / name).read_bytes() for name in _PIPELINE_SOURCES)


TEXT_PIPELINE_VERSION = pipeline_version(_read_pipeline_sources())
