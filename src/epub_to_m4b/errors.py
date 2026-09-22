"""Base class for user-facing errors.

Anything that subclasses :class:`EpubToM4bError` describes a problem the user
can act on (bad config, TTS request that failed for good, ffmpeg failure). The
CLI catches this base, prints ``error: <message>`` to stderr and exits 1;
every other exception is a bug and keeps its traceback.
"""

from __future__ import annotations


class EpubToM4bError(Exception):
    """A problem the user can fix; printed as ``error: ...`` by the CLI."""
