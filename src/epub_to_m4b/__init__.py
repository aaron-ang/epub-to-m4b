"""Turn an EPUB into an M4B audiobook with a WebVTT transcript."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("epub-to-m4b")
except PackageNotFoundError:
    # Running from a source checkout that was never installed, so there is no
    # dist-info to read; PEP 440 local-version marker instead of a stale literal.
    __version__ = "0+unknown"
