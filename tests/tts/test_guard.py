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


def test_short_limit_seconds_formula() -> None:
    policy = guard.RunawayPolicy(min_seconds_per_char=0.05, min_chars=10)
    assert guard.short_limit_seconds("a" * 40, policy) == 0.05 * 40


def test_short_limit_is_zero_below_min_chars() -> None:
    policy = guard.RunawayPolicy(min_chars=10)
    assert guard.short_limit_seconds("a" * 9, policy) == 0.0


def test_window_miss_is_zero_inside_and_distance_outside() -> None:
    text = "a" * 40
    short, long = guard.short_limit_seconds(text, _POLICY), _retry(text)
    assert guard.window_miss_seconds(_clip((short + long) / 2), text, _POLICY) == 0.0
    assert guard.window_miss_seconds(_clip(long + 1.0), text, _POLICY) == pytest.approx(
        1.0, abs=1e-3
    )
    assert guard.window_miss_seconds(_clip(short - 0.5), text, _POLICY) == pytest.approx(
        0.5, abs=1e-3
    )


def test_clip_miss_capped_over_retry_limit_is_infinite() -> None:
    text = "x" * 10
    over = _clip(_retry(text) + 0.5)
    assert guard.clip_miss_seconds(over, text, _POLICY, capped=True) == math.inf
    assert guard.clip_miss_seconds(over, text, _POLICY, capped=False) == pytest.approx(
        0.5, abs=1e-3
    )


def test_clip_miss_short_capped_clip_is_finite() -> None:
    text = "s" * 120
    short = guard.short_limit_seconds(text, _POLICY)
    assert guard.clip_miss_seconds(_clip(short - 1.0), text, _POLICY, capped=True) == pytest.approx(
        1.0, abs=1e-3
    )


def test_clip_miss_inside_window_is_zero() -> None:
    text = "a" * 40
    assert guard.clip_miss_seconds(_clip(3.0), text, _POLICY, capped=True) == 0.0
