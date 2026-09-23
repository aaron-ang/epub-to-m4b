from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import pytest

from epub_to_m4b.book import AudioClip
from epub_to_m4b.tts import guard

_SAMPLE_RATE = 24000


def _clip(seconds: float, sample_rate: int = _SAMPLE_RATE, value: float = 1.0) -> AudioClip:
    n = int(seconds * sample_rate)
    return AudioClip(samples=np.full(n, value, dtype=np.float32), sample_rate=sample_rate)


_POLICY = guard.DEFAULT_RUNAWAY_POLICY
_FRAME_RATE = 12.5


def _retry(text: str) -> float:
    return guard.retry_limit_seconds(text, _POLICY)


def _cap(texts: Sequence[str]) -> float:
    return guard.cap_seconds(texts, _POLICY)


def test_retry_limit_seconds_formula() -> None:
    policy = guard.RunawayPolicy(base_seconds=1.5, seconds_per_char=0.3)
    assert guard.retry_limit_seconds("hello", policy) == 1.5 + 0.3 * 5


def test_cap_seconds_is_longest_retry_limit_times_slack() -> None:
    policy = guard.RunawayPolicy(cap_slack=1.5)
    texts = ["short", "a fair bit longer than short"]
    expected = max(guard.retry_limit_seconds(t, policy) for t in texts) * 1.5
    assert guard.cap_seconds(texts, policy) == expected


def test_cap_leaves_room_above_retry_limit() -> None:
    for text in ["", "a" * 125]:
        assert _cap([text]) > _retry(text)


def test_max_new_tokens_matches_formula() -> None:
    texts = ["short", "a fair bit longer than short"]
    expected = math.ceil(_cap(texts) * _FRAME_RATE)
    assert guard.max_new_tokens(texts, _POLICY, _FRAME_RATE) == expected


def test_max_new_tokens_scales_with_frame_rate() -> None:
    texts = ["hello there"]
    assert guard.max_new_tokens(texts, _POLICY, 25.0) > guard.max_new_tokens(
        texts, _POLICY, _FRAME_RATE
    )


def test_max_new_tokens_covers_cap_seconds() -> None:
    for text in ["", "a", "a" * 60, "a" * 125, "a" * 200]:
        assert guard.max_new_tokens([text], _POLICY, _FRAME_RATE) / _FRAME_RATE >= _cap([text])


def test_max_new_tokens_scales_with_longest_text_in_batch() -> None:
    short_batch = ["hi"]
    long_batch = ["hi", "a much, much longer sentence than the short one in the other batch"]
    assert guard.max_new_tokens(long_batch, _POLICY, _FRAME_RATE) > guard.max_new_tokens(
        short_batch, _POLICY, _FRAME_RATE
    )


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


class _FakeReseed:
    """Batched reseed stand-in: records ``(texts, attempt)`` per call and answers
    each text with a clip whose length comes from ``lengths[attempt][text]``."""

    def __init__(self, lengths: dict[int, dict[str, float]]) -> None:
        self.lengths = lengths
        self.calls: list[tuple[list[str], int]] = []

    def __call__(self, texts: Sequence[str], attempt: int) -> list[AudioClip]:
        self.calls.append((list(texts), attempt))
        return [_clip(self.lengths[attempt][text]) for text in texts]


def test_apply_guard_keeps_clip_under_retry_limit_as_is_no_retry() -> None:
    text = "A short sentence."
    clip = _clip(0.5)  # well under 2.0 + 0.1*len(text)
    reseed = _FakeReseed({})

    clips, notes = guard.apply_guard([clip], [text], reseed)
    assert clips == [clip]
    assert clips[0] is clip
    assert notes == []
    assert reseed.calls == []


def test_apply_guard_passes_only_runaway_texts_in_index_order() -> None:
    ok_a, run_b, ok_c, run_d = "aaaaa", "b" * 10, "ccccc", "d" * 10
    limit_b = _retry(run_b)
    limit_d = _retry(run_d)
    clips_in = [_clip(0.1), _clip(limit_b + 1.0), _clip(0.1), _clip(limit_d + 1.0)]
    reseed = _FakeReseed({1: {run_b: limit_b - 0.1, run_d: limit_d - 0.1}})

    clips, notes = guard.apply_guard(clips_in, [ok_a, run_b, ok_c, run_d], reseed)
    assert reseed.calls == [([run_b, run_d], 1)]
    assert clips[0] is clips_in[0]
    assert clips[2] is clips_in[2]
    assert clips[1].seconds == limit_b - 0.1
    assert clips[3].seconds == limit_d - 0.1
    assert len(notes) == 2
    assert all("recovered" in note for note in notes)
    assert repr(run_b) in notes[0] and repr(run_d) in notes[1]


def test_apply_guard_drops_recovered_texts_from_next_attempt() -> None:
    run_b, run_d = "b" * 10, "d" * 10
    limit = _retry(run_b)
    clips_in = [_clip(limit + 1.0), _clip(limit + 1.0)]
    reseed = _FakeReseed(
        {
            1: {run_b: limit - 0.1, run_d: limit + 0.5},  # b recovers, d still over
            2: {run_d: limit - 0.2},
        }
    )

    clips, notes = guard.apply_guard(clips_in, [run_b, run_d], reseed)
    assert reseed.calls == [([run_b, run_d], 1), ([run_d], 2)]
    assert clips[0].seconds == limit - 0.1
    assert clips[1].seconds == limit - 0.2
    assert len(notes) == 2
    assert all("recovered" in note for note in notes)


def test_apply_guard_between_limits_kept_with_note() -> None:
    text = "x" * 10
    retry_limit = _retry(text)
    cut_limit = _cap([text])
    over_retry_under_cut = (retry_limit + cut_limit) / 2
    clip = _clip(over_retry_under_cut)
    # every retry comes back just as long - never recovers
    reseed = _FakeReseed({a: {text: over_retry_under_cut} for a in range(1, 10)})

    clips, notes = guard.apply_guard([clip], [text], reseed)
    assert len(clips) == 1
    assert clips[0].seconds == over_retry_under_cut
    assert [attempt for _, attempt in reseed.calls] == list(range(1, _POLICY.retries + 1))
    assert notes == [
        f"slow clip kept at {over_retry_under_cut:.1f}s after {_POLICY.retries} retries: "
        f"{text[:60]!r}"
    ]


def test_apply_guard_cuts_and_fades_when_still_over_cut_limit_after_retries() -> None:
    text = "y" * 10
    cut_limit = _cap([text])
    clip = _clip(cut_limit + 5.0)
    # every retry also runs away, just as long
    reseed = _FakeReseed({a: {text: cut_limit + 5.0} for a in range(1, 10)})

    clips, notes = guard.apply_guard([clip], [text], reseed)
    assert len(clips) == 1
    guarded = clips[0]
    assert len(guarded.samples) == int(cut_limit * guarded.sample_rate)
    assert guarded.samples[-1] == 0.0  # fade reaches ~0
    assert len(reseed.calls) == _POLICY.retries
    assert notes == [
        f"runaway clip cut to {cut_limit:.1f}s after {_POLICY.retries} retries: {text[:60]!r}"
    ]


def test_apply_guard_retry_succeeds_partway_stops_early() -> None:
    text = "z" * 10
    retry_limit = _retry(text)
    clip = _clip(retry_limit + 1.0)  # over retry limit, triggers retries
    reseed = _FakeReseed({1: {text: retry_limit + 0.5}, 2: {text: retry_limit - 0.5}})

    clips, notes = guard.apply_guard([clip], [text], reseed)
    assert reseed.calls == [([text], 1), ([text], 2)]
    assert clips[0].seconds == retry_limit - 0.5
    assert notes == [f"runaway clip recovered with a reseed: {text[:60]!r}"]


def test_apply_guard_stops_after_first_successful_retry() -> None:
    text = "w" * 10
    retry_limit = _retry(text)
    clip = _clip(retry_limit + 1.0)
    reseed = _FakeReseed({1: {text: retry_limit - 0.1}})  # succeeds immediately

    clips, _notes = guard.apply_guard([clip], [text], reseed)
    assert reseed.calls == [([text], 1)]  # never tries attempt 2
    assert clips[0].seconds == retry_limit - 0.1


def test_apply_guard_keeps_shortest_result_across_retries() -> None:
    text = "v" * 10
    cut_limit = _cap([text])
    clip = _clip(cut_limit + 10.0)
    # attempt 1 comes back worse than the original clip; attempt 2 is the
    # best (shortest) of all, still over cut_limit
    reseed = _FakeReseed({1: {text: cut_limit + 20.0}, 2: {text: cut_limit + 1.0}})

    clips, _notes = guard.apply_guard([clip], [text], reseed)
    # cut applied to the shortest candidate seen (cut_limit + 1.0), not the
    # original or the worse retry
    assert len(clips[0].samples) == int(cut_limit * clips[0].sample_rate)


def test_apply_guard_keeps_shortest_even_when_still_over_limit_and_not_cut() -> None:
    text = "u" * 10
    retry_limit = _retry(text)
    cut_limit = _cap([text])
    clip = _clip(cut_limit + 3.0)
    between = (retry_limit + cut_limit) / 2
    reseed = _FakeReseed({1: {text: cut_limit + 9.0}, 2: {text: between}})

    clips, notes = guard.apply_guard([clip], [text], reseed)
    assert clips[0].seconds == between
    assert len(notes) == 1 and "kept" in notes[0]


def test_apply_guard_raises_on_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match="clips"):
        guard.apply_guard([_clip(1.0)], ["a", "b"], _FakeReseed({}))


def test_apply_guard_raises_when_reseed_returns_wrong_count() -> None:
    text = "t" * 10
    clip = _clip(_retry(text) + 1.0)

    def reseed(texts: Sequence[str], attempt: int) -> list[AudioClip]:
        return []

    with pytest.raises(ValueError, match="reseed"):
        guard.apply_guard([clip], [text], reseed)


def test_apply_guard_multiple_clips_independent() -> None:
    text_ok = "short"
    text_runaway = "r" * 10
    retry_limit = _retry(text_runaway)
    clip_ok = _clip(0.1)
    clip_runaway = _clip(retry_limit + 1.0)
    reseed = _FakeReseed({1: {text_runaway: retry_limit - 0.1}})

    clips, notes = guard.apply_guard([clip_ok, clip_runaway], [text_ok, text_runaway], reseed)
    assert clips[0] is clip_ok
    assert clips[1].seconds == retry_limit - 0.1
    assert len(notes) == 1


def test_apply_guard_cuts_to_batch_cap_not_own_limit() -> None:
    short, long = "s" * 5, "l" * 100
    batch_cap = _cap([short, long])
    assert batch_cap > _cap([short])
    runaway = batch_cap + 3.0
    reseed = _FakeReseed({a: {short: runaway} for a in range(1, 10)})

    clips, notes = guard.apply_guard([_clip(runaway), _clip(0.1)], [short, long], reseed)
    assert len(clips[0].samples) == int(batch_cap * clips[0].sample_rate)
    assert notes == [
        f"runaway clip cut to {batch_cap:.1f}s after {_POLICY.retries} retries: {short[:60]!r}"
    ]


def test_apply_guard_over_retry_but_under_batch_cap_is_kept() -> None:
    short, long = "s" * 5, "l" * 100
    slow = _cap([short]) + 1.0  # past its own cap, still inside the batch's
    reseed = _FakeReseed({a: {short: slow} for a in range(1, 10)})

    clips, notes = guard.apply_guard([_clip(slow), _clip(0.1)], [short, long], reseed)
    assert clips[0].seconds == slow
    assert len(notes) == 1 and "slow clip kept" in notes[0]


def test_apply_guard_retries_come_from_policy() -> None:
    text = "q" * 10
    policy = guard.RunawayPolicy(retries=4)
    stuck = guard.retry_limit_seconds(text, policy) + 0.5
    reseed = _FakeReseed({a: {text: stuck} for a in range(1, 10)})

    _clips, notes = guard.apply_guard([_clip(stuck)], [text], reseed, policy)
    assert [attempt for _, attempt in reseed.calls] == [1, 2, 3, 4]
    assert "after 4 retries" in notes[0]


def test_apply_guard_cut_limit_respects_max_cap_seconds() -> None:
    text = "m" * 10
    ceiling = (_retry(text) + _cap([text])) / 2
    runaway = _cap([text]) + 1.0
    reseed = _FakeReseed({a: {text: runaway} for a in range(1, 10)})

    clips, notes = guard.apply_guard(
        [_clip(runaway)], [text], reseed, _POLICY, max_cap_seconds=ceiling
    )
    assert len(clips[0].samples) == int(ceiling * clips[0].sample_rate)
    assert notes[0].startswith(f"runaway clip cut to {ceiling:.1f}s")
