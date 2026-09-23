"""Pure runaway-clip budget math: retry limit, batch cap, cut+fade.

No HTTP or subprocess here - this is the arithmetic that decides whether a
synthesized clip is plausibly real speech, worth a reseed, or bad enough to
truncate. Keeping it pure (plain floats/arrays in, plain floats/arrays out)
is what makes it exhaustively testable without a GPU or a running sidecar;
``tts/breeze.py`` wires it to HTTP retries.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from epub_to_m4b.book import AudioClip


@dataclass(frozen=True)
class RunawayPolicy:
    """Duration budget that decides whether a clip is real speech or a runaway.

    A clip over its retry limit (fixed lead-in plus a per-character
    allowance) is reseeded up to ``retries`` times. The server-side token cap
    is the batch's longest retry limit times ``cap_slack``: every clip in a
    batch decodes in lockstep, so one runaway holds the whole batch until the
    cap, and keeping it just above the retry limit bounds that stall. A clip
    stopped by the cap is over its retry limit, so it is reseeded; a survivor
    still longer than the cap is cut there and faded, so it never ends
    mid-word.
    """

    base_seconds: float = 2.0
    seconds_per_char: float = 0.10
    cap_slack: float = 1.25
    retries: int = 2


DEFAULT_RUNAWAY_POLICY = RunawayPolicy()

# Fade-out on a cut clip: long enough to avoid an audible click, short
# enough not to swallow the last syllable.
FADE_SECONDS = 0.02


def retry_limit_seconds(text: str, policy: RunawayPolicy) -> float:
    """A clip longer than this is retried with a fresh seed."""
    return policy.base_seconds + policy.seconds_per_char * len(text)


def cap_seconds(texts: Sequence[str], policy: RunawayPolicy) -> float:
    """Server-side generation cap for a batch, in seconds of audio.

    Sized off the batch's longest text so one runaway doesn't stall the
    others past its own budget. Also the cut limit: a clip still over its
    retry limit after reseeding is truncated here.
    """
    return max(retry_limit_seconds(text, policy) for text in texts) * policy.cap_slack


def max_new_tokens(texts: Sequence[str], policy: RunawayPolicy, frame_rate: float) -> int:
    """``cap_seconds`` as codec frames, the unit the server counts in."""
    return math.ceil(cap_seconds(texts, policy) * frame_rate)


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
    policy: RunawayPolicy = DEFAULT_RUNAWAY_POLICY,
    *,
    max_cap_seconds: float = math.inf,
) -> tuple[list[AudioClip], list[str]]:
    """Retry runaway clips with fresh seeds, then cut+fade anything still too long.

    Reseeding is batched: ``reseed_fn(texts, attempt)`` is called once per
    ``attempt`` in ``1..policy.retries`` with only the texts whose best clip
    so far is still over the retry limit, and must return one clip per text
    in the same order. The caller's closure decides what the attempt number
    means as an actual seed. For each text the shortest candidate seen wins,
    and a text leaves the retry set once its best is under the limit; the
    loop stops early when the set empties. One round trip per attempt for the
    whole batch is what keeps several runaways from each paying the full
    fixed cost of a separate request.

    A text still over its retry limit afterwards is cut and faded to the
    batch's ``cap_seconds`` if longer than it, else kept as a slow clip.
    ``max_cap_seconds`` is the server's own ceiling on that cap, for a server
    that clamps the requested token count.

    Returns the guarded clips alongside human-readable notes for anything
    that needed a retry or a cut - diagnostic signal a real user would want,
    not silently swallowed.
    """
    if len(clips) != len(texts):
        raise ValueError(f"got {len(clips)} clips for {len(texts)} texts")

    best = list(clips)
    limits = [retry_limit_seconds(text, policy) for text in texts]
    runaway = [i for i, clip in enumerate(clips) if clip.seconds > limits[i]]
    flagged = list(runaway)

    for attempt in range(1, policy.retries + 1):
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
    cut_limit = min(cap_seconds(texts, policy), max_cap_seconds) if flagged else 0.0
    for i in flagged:
        text = texts[i]
        clip = best[i]
        if clip.seconds > cut_limit:
            cut_samples = cut_and_fade(clip.samples, clip.sample_rate, cut_limit)
            best[i] = AudioClip(samples=cut_samples, sample_rate=clip.sample_rate)
            notes.append(
                f"runaway clip cut to {cut_limit:.1f}s after {policy.retries} retries: "
                f"{text[:60]!r}"
            )
        elif clip.seconds > limits[i]:
            notes.append(
                f"slow clip kept at {clip.seconds:.1f}s after {policy.retries} retries: "
                f"{text[:60]!r}"
            )
        else:
            notes.append(f"runaway clip recovered with a reseed: {text[:60]!r}")

    return best, notes
