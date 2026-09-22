"""Pure runaway-clip budget math: retry/cut limits, token cap, cut+fade.

No HTTP or subprocess here - this is the arithmetic that decides whether a
synthesized clip is plausibly real speech, worth a reseed, or bad enough to
truncate. Keeping it pure (plain floats/arrays in, plain floats/arrays out)
is what makes it exhaustively testable without a GPU or a running sidecar;
``tts/breeze.py`` is the only current caller, wiring this to actual HTTP
retries.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

import numpy as np
import numpy.typing as npt

from epub_to_m4b.book import AudioClip

# Clip budget: fixed lead-in plus a per-character allowance. The retry limit
# sits just above the slowest genuine narration so short babble still
# triggers a reseed; the cut limit is far above it so only an unmistakable
# runaway is truncated and slow-but-real speech is kept whole.
CLIP_BASE_SECONDS = 2.0
CLIP_SECONDS_PER_CHAR = 0.10
CUT_SECONDS_PER_CHAR = 0.20
# Audio tokens/sec of the codec Breeze decodes with.
TOKENS_PER_SECOND = 12.5
# The server-side cap is loose on purpose: it only stops a batch from
# stalling on one runaway, while the client-side duration check (this
# module) does the precise retry/cut.
TOKEN_CAP_SLACK = 2.0
FADE_SECONDS = 0.02
RUNAWAY_RETRIES = 2


def retry_limit_seconds(text: str) -> float:
    """A clip longer than this is retried with a fresh seed."""
    return CLIP_BASE_SECONDS + CLIP_SECONDS_PER_CHAR * len(text)


def cut_limit_seconds(text: str) -> float:
    """A clip still over this after retries gets truncated."""
    return CLIP_BASE_SECONDS + CUT_SECONDS_PER_CHAR * len(text)


def max_new_tokens(texts: Sequence[str]) -> int:
    """Server-side generation cap for a batch.

    Sized off the batch's longest text so one runaway doesn't stall the
    others past its own budget.
    """
    longest = max(retry_limit_seconds(text) for text in texts)
    return math.ceil(longest * TOKEN_CAP_SLACK * TOKENS_PER_SECOND)


def cut_and_fade(
    samples: npt.NDArray[np.float32], sample_rate: int, limit_seconds: float
) -> npt.NDArray[np.float32]:
    """Truncate to ``limit_seconds`` with a linear fade-out so the cut isn't a click."""
    cut = samples[: int(limit_seconds * sample_rate)].copy()
    fade_samples = min(len(cut), int(FADE_SECONDS * sample_rate))
    if fade_samples:
        cut[-fade_samples:] *= np.linspace(1.0, 0.0, fade_samples, dtype=np.float32)
    return cut


def apply_guard(
    clips: Sequence[AudioClip],
    texts: Sequence[str],
    reseed_fn: Callable[[Sequence[str], int], list[AudioClip]],
) -> tuple[list[AudioClip], list[str]]:
    """Retry runaway clips with fresh seeds, then cut+fade anything still too long.

    Reseeding is batched: ``reseed_fn(texts, attempt)`` is called once per
    ``attempt`` in ``1..RUNAWAY_RETRIES`` with only the texts whose best clip
    so far is still over the retry limit, and must return one clip per text
    in the same order. The caller's closure decides what the attempt number
    means as an actual seed. For each text the shortest candidate seen wins,
    and a text leaves the retry set once its best is under the limit; the
    loop stops early when the set empties. One round trip per attempt for the
    whole batch is what keeps several runaways from each paying the full
    fixed cost of a separate request.

    Returns the guarded clips alongside human-readable notes for anything
    that needed a retry or a cut - diagnostic signal a real user would want,
    not silently swallowed.
    """
    if len(clips) != len(texts):
        raise ValueError(f"got {len(clips)} clips for {len(texts)} texts")

    best = list(clips)
    limits = [retry_limit_seconds(text) for text in texts]
    runaway = [i for i, clip in enumerate(clips) if clip.seconds > limits[i]]
    flagged = list(runaway)

    for attempt in range(1, RUNAWAY_RETRIES + 1):
        if not runaway:
            break
        candidates = reseed_fn([texts[i] for i in runaway], attempt)
        if len(candidates) != len(runaway):
            raise ValueError(f"reseed returned {len(candidates)} clips for {len(runaway)} texts")
        for i, candidate in zip(runaway, candidates, strict=True):
            if candidate.seconds < best[i].seconds:
                best[i] = candidate
        runaway = [i for i in runaway if best[i].seconds > limits[i]]

    notes: list[str] = []
    for i in flagged:
        text = texts[i]
        clip = best[i]
        cut_limit = cut_limit_seconds(text)
        if clip.seconds > cut_limit:
            cut_samples = cut_and_fade(clip.samples, clip.sample_rate, cut_limit)
            best[i] = AudioClip(samples=cut_samples, sample_rate=clip.sample_rate)
            notes.append(
                f"runaway clip cut to {cut_limit:.1f}s after {RUNAWAY_RETRIES} retries: "
                f"{text[:60]!r}"
            )
        elif clip.seconds > limits[i]:
            notes.append(
                f"slow clip kept at {clip.seconds:.1f}s after {RUNAWAY_RETRIES} retries: "
                f"{text[:60]!r}"
            )
        else:
            notes.append(f"runaway clip recovered with a reseed: {text[:60]!r}")

    return best, notes
