from __future__ import annotations

import math

import numpy as np
import pytest

from epub_to_m4b.book import AudioClip
from epub_to_m4b.tts import guard

_SAMPLE_RATE = 24000


def _clip(seconds: float, sample_rate: int = _SAMPLE_RATE, value: float = 1.0) -> AudioClip:
    n = int(seconds * sample_rate)
    return AudioClip(samples=np.full(n, value, dtype=np.float32), sample_rate=sample_rate)


def test_retry_limit_seconds_formula() -> None:
    text = "hello"  # 5 chars
    expected = guard.CLIP_BASE_SECONDS + guard.CLIP_SECONDS_PER_CHAR * 5
    assert guard.retry_limit_seconds(text) == expected


def test_cut_limit_seconds_formula() -> None:
    text = "hello"
    expected = guard.CLIP_BASE_SECONDS + guard.CUT_SECONDS_PER_CHAR * 5
    assert guard.cut_limit_seconds(text) == expected


def test_cut_limit_always_at_least_retry_limit() -> None:
    for text in ["", "a", "a" * 200]:
        assert guard.cut_limit_seconds(text) >= guard.retry_limit_seconds(text)


def test_max_new_tokens_scales_with_longest_text_in_batch() -> None:
    short_batch = ["hi"]
    long_batch = ["hi", "a much, much longer sentence than the short one in the other batch"]
    assert guard.max_new_tokens(long_batch) > guard.max_new_tokens(short_batch)


def test_max_new_tokens_matches_formula() -> None:
    texts = ["short", "a fair bit longer than short"]
    longest_limit = max(guard.retry_limit_seconds(t) for t in texts)
    expected = math.ceil(longest_limit * guard.TOKEN_CAP_SLACK * guard.TOKENS_PER_SECOND)
    assert guard.max_new_tokens(texts) == expected


def test_cut_and_fade_truncates_to_exact_sample_count() -> None:
    clip = _clip(5.0)
    limit = 2.0
    cut = guard.cut_and_fade(clip.samples, clip.sample_rate, limit)
    assert len(cut) == int(limit * clip.sample_rate)


def test_cut_and_fade_tapers_to_near_zero_at_the_end() -> None:
    clip = _clip(5.0, value=1.0)
    cut = guard.cut_and_fade(clip.samples, clip.sample_rate, 2.0)
    assert cut[-1] == 0.0
    assert cut[-1] < cut[-int(guard.FADE_SECONDS * clip.sample_rate)]


def test_cut_and_fade_handles_clip_shorter_than_fade_window() -> None:
    # limit_seconds bigger than the fade window itself once truncated small
    clip = _clip(0.005, value=1.0)  # shorter than FADE_SECONDS
    cut = guard.cut_and_fade(clip.samples, clip.sample_rate, 1.0)
    # whole (short) clip used as the fade window; still tapers to zero
    assert cut[-1] == 0.0
    assert len(cut) == len(clip.samples)


def test_apply_guard_keeps_clip_under_retry_limit_as_is_no_retry() -> None:
    text = "A short sentence."
    clip = _clip(0.5)  # well under 2.0 + 0.1*len(text)

    def reseed(_text: str, _attempt: int) -> AudioClip:
        raise AssertionError("reseed_fn should not be called for a clip under the retry limit")

    clips, notes = guard.apply_guard([clip], [text], reseed)
    assert clips == [clip]
    assert notes == []


def test_apply_guard_between_limits_kept_with_note() -> None:
    text = "x" * 10  # retry limit = 3.0s, cut limit = 4.0s
    retry_limit = guard.retry_limit_seconds(text)
    cut_limit = guard.cut_limit_seconds(text)
    over_retry_under_cut = (retry_limit + cut_limit) / 2
    clip = _clip(over_retry_under_cut)

    calls = []

    def reseed(reseed_text: str, attempt: int) -> AudioClip:
        calls.append(attempt)
        # retries come back just as long - never recovers
        return _clip(over_retry_under_cut)

    clips, notes = guard.apply_guard([clip], [text], reseed)
    assert len(clips) == 1
    assert clips[0].seconds == over_retry_under_cut
    assert len(calls) == guard.RUNAWAY_RETRIES
    assert len(notes) == 1
    assert "kept" in notes[0]


def test_apply_guard_cuts_and_fades_when_still_over_cut_limit_after_retries() -> None:
    text = "y" * 10
    cut_limit = guard.cut_limit_seconds(text)
    clip = _clip(cut_limit + 5.0)

    def reseed(reseed_text: str, attempt: int) -> AudioClip:
        # every retry also runs away, just as long
        return _clip(cut_limit + 5.0)

    clips, notes = guard.apply_guard([clip], [text], reseed)
    assert len(clips) == 1
    guarded = clips[0]
    assert len(guarded.samples) == int(cut_limit * guarded.sample_rate)
    assert guarded.samples[-1] == 0.0  # fade reaches ~0
    assert len(notes) == 1
    assert "cut" in notes[0]


def test_apply_guard_retry_succeeds_partway_stops_early() -> None:
    text = "z" * 10
    retry_limit = guard.retry_limit_seconds(text)
    clip = _clip(retry_limit + 1.0)  # over retry limit, triggers retries

    calls = []

    def reseed(reseed_text: str, attempt: int) -> AudioClip:
        calls.append(attempt)
        if attempt == 1:
            return _clip(retry_limit + 0.5)  # still too long
        return _clip(retry_limit - 0.5)  # attempt 2: recovers

    clips, notes = guard.apply_guard([clip], [text], reseed)
    assert calls == [1, 2]  # stops after attempt 2 succeeds, no further attempts
    assert clips[0].seconds == retry_limit - 0.5
    assert len(notes) == 1
    assert "recovered" in notes[0]


def test_apply_guard_stops_after_first_successful_retry() -> None:
    text = "w" * 10
    retry_limit = guard.retry_limit_seconds(text)
    clip = _clip(retry_limit + 1.0)

    calls = []

    def reseed(reseed_text: str, attempt: int) -> AudioClip:
        calls.append(attempt)
        return _clip(retry_limit - 0.1)  # succeeds immediately on attempt 1

    clips, _notes = guard.apply_guard([clip], [text], reseed)
    assert calls == [1]  # never tries attempt 2
    assert clips[0].seconds == retry_limit - 0.1


def test_apply_guard_keeps_shortest_result_across_retries() -> None:
    text = "v" * 10
    cut_limit = guard.cut_limit_seconds(text)
    clip = _clip(cut_limit + 10.0)

    def reseed(reseed_text: str, attempt: int) -> AudioClip:
        # attempt 1 comes back worse than the original clip; attempt 2 is
        # the best (shortest) of all, still over cut_limit
        if attempt == 1:
            return _clip(cut_limit + 20.0)
        return _clip(cut_limit + 1.0)

    clips, _notes = guard.apply_guard([clip], [text], reseed)
    # cut applied to the shortest candidate seen (cut_limit + 1.0), not the
    # original or the worse retry
    assert len(clips[0].samples) == int(cut_limit * clips[0].sample_rate)


def test_apply_guard_raises_on_mismatched_lengths() -> None:

    with pytest.raises(ValueError, match="clips"):
        guard.apply_guard([_clip(1.0)], ["a", "b"], lambda t, a: _clip(1.0))


def test_apply_guard_multiple_clips_independent() -> None:
    text_ok = "short"
    text_runaway = "r" * 10
    retry_limit = guard.retry_limit_seconds(text_runaway)
    clip_ok = _clip(0.1)
    clip_runaway = _clip(retry_limit + 1.0)

    def reseed(text: str, attempt: int) -> AudioClip:
        return _clip(retry_limit - 0.1)

    clips, notes = guard.apply_guard([clip_ok, clip_runaway], [text_ok, text_runaway], reseed)
    assert clips[0] is clip_ok
    assert clips[1].seconds == retry_limit - 0.1
    assert len(notes) == 1
