"""Pure runaway-clip budget math: duration window, batch cap, clip scoring.

No HTTP or subprocess here - this is the arithmetic that decides whether a
synthesized clip is plausibly real speech or worth a reseed. Keeping it pure
(plain floats/arrays in, plain floats/arrays out) is what makes it
exhaustively testable without a GPU or a running sidecar; ``tts/breeze.py``
wires it to HTTP and ``synth/retry.py`` batches the reseeds. Nothing here
ever truncates audio: every clip that reaches the book is one the model
finished on its own.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from epub_to_m4b.book import AudioClip


@dataclass(frozen=True)
class RunawayPolicy:
    """Duration window that decides whether a clip is real speech.

    A clip over its retry limit (fixed lead-in plus a per-character
    allowance) ran away; a clip of at least ``min_chars`` characters under
    ``min_seconds_per_char`` per character stopped early and dropped words.
    Either is reseeded up to ``retries`` times, in full batches queued by
    ``synth/retry.py``. The first pass's server-side
    token cap is the batch's longest retry limit times ``cap_slack``: every
    clip in a batch decodes in lockstep, so one runaway holds the whole batch
    until the cap, and keeping it just above the retry limit bounds that
    stall. Reseeds get the same cap. Real speech never reaches its retry
    limit (the longest verified clip in a whole book came to 0.98 of it), so
    a clip stopped by the cap is babble: it scores infinite and loses to any
    complete take. Only a text whose every take hit the cap gets one last
    uncapped reseed, so the kept clip is never one the cap truncated.
    """

    base_seconds: float = 2.0
    seconds_per_char: float = 0.10
    min_seconds_per_char: float = 0.045
    min_chars: int = 20
    cap_slack: float = 1.25
    retries: int = 2


DEFAULT_RUNAWAY_POLICY = RunawayPolicy()


def retry_limit_seconds(text: str, policy: RunawayPolicy) -> float:
    """A clip longer than this is retried with a fresh seed."""
    return policy.base_seconds + policy.seconds_per_char * len(text)


def short_limit_seconds(text: str, policy: RunawayPolicy) -> float:
    """A clip shorter than this is retried with a fresh seed.

    Zero below ``min_chars``: a few words have too little text for their
    length to say anything about dropped speech.
    """
    if len(text) < policy.min_chars:
        return 0.0
    return policy.min_seconds_per_char * len(text)


def window_miss_seconds(clip: AudioClip, text: str, policy: RunawayPolicy) -> float:
    """How far ``clip`` falls outside its duration window; 0.0 inside it."""
    short = short_limit_seconds(text, policy)
    long = retry_limit_seconds(text, policy)
    return max(short - clip.seconds, clip.seconds - long, 0.0)


def cap_seconds(texts: Sequence[str], policy: RunawayPolicy) -> float:
    """First-pass server-side generation cap for a batch, in seconds of audio.

    Sized off the batch's longest text so one runaway doesn't stall the
    others past its own budget.
    """
    return max(retry_limit_seconds(text, policy) for text in texts) * policy.cap_slack


def max_new_tokens(texts: Sequence[str], policy: RunawayPolicy, frame_rate: float) -> int:
    """``cap_seconds`` as codec frames, the unit the server counts in."""
    return math.ceil(cap_seconds(texts, policy) * frame_rate)


def clip_miss_seconds(clip: AudioClip, text: str, policy: RunawayPolicy, *, capped: bool) -> float:
    """Score ``clip`` for the retry queue: 0.0 keeps it, lower is better.

    The batch cap sits above every text's retry limit, so a capped clip over
    its retry limit may be the one the cap stopped mid-word: it scores
    infinite and any complete take replaces it.
    """
    if capped and clip.seconds > retry_limit_seconds(text, policy):
        return math.inf
    return window_miss_seconds(clip, text, policy)
