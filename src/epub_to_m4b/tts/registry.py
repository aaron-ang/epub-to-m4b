"""Engine name -> factory lookup used by the CLI's ``--engine`` flag.

Only the fake engines exist as of this milestone. Real engines register the
same way once they land: add the class here (or, for third-party engines,
via an ``epub_to_m4b.engines`` entry point - not needed yet with a single
package).
"""

from __future__ import annotations

from collections.abc import Callable

from epub_to_m4b.tts.base import TTSEngine
from epub_to_m4b.tts.fake import SilenceEngine, ToneEngine

_ENGINES: dict[str, Callable[[], TTSEngine]] = {
    SilenceEngine.name: SilenceEngine,
    ToneEngine.name: ToneEngine,
}


def available_engines() -> list[str]:
    return sorted(_ENGINES)


def create_engine(name: str) -> TTSEngine:
    try:
        factory = _ENGINES[name]
    except KeyError:
        raise ValueError(f"unknown engine: {name!r}") from None
    return factory()
