from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from email.utils import format_datetime

import httpx
import numpy as np
import pytest

from epub_to_m4b.tts.http import RetryPolicy, TTSError, pcm_response_to_clip, post_with_retry

_URL = "https://tts.example/v1/speak"


def _sequence(*responses: httpx.Response | Exception) -> tuple[httpx.MockTransport, list[int]]:
    """MockTransport that replays ``responses`` in order, one per request,
    counting requests. An Exception entry is raised instead of returned."""
    queue: Iterator[httpx.Response | Exception] = iter(responses)
    seen: list[int] = [0]

    def handler(_request: httpx.Request) -> httpx.Response:
        seen[0] += 1
        item = next(queue)
        if isinstance(item, Exception):
            raise item
        return item

    return httpx.MockTransport(handler), seen


def test_success_on_first_try_does_not_sleep() -> None:
    transport, seen = _sequence(httpx.Response(200, content=b"ok"))
    sleeps: list[float] = []
    with httpx.Client(transport=transport) as client:
        response = post_with_retry(client, _URL, sleep=sleeps.append, json={"text": "hi"})
    assert response.status_code == 200
    assert response.content == b"ok"
    assert seen[0] == 1
    assert sleeps == []


def test_429_with_retry_after_seconds_waits_exactly_that_long() -> None:
    transport, seen = _sequence(
        httpx.Response(429, headers={"Retry-After": "3"}), httpx.Response(200, content=b"ok")
    )
    sleeps: list[float] = []
    with httpx.Client(transport=transport) as client:
        response = post_with_retry(client, _URL, sleep=sleeps.append)
    assert response.status_code == 200
    assert seen[0] == 2
    assert sleeps == [3.0]


def test_retry_after_http_date_is_parsed_relative_to_now() -> None:
    when = datetime.now(UTC) + timedelta(seconds=5)
    transport, _seen = _sequence(
        httpx.Response(503, headers={"Retry-After": format_datetime(when, usegmt=True)}),
        httpx.Response(200, content=b"ok"),
    )
    sleeps: list[float] = []
    with httpx.Client(transport=transport) as client:
        post_with_retry(client, _URL, sleep=sleeps.append)
    assert len(sleeps) == 1
    # HTTP-dates carry whole seconds, so the wait lands anywhere in a
    # roughly two-second window around the nominal value.
    assert 3.0 <= sleeps[0] <= 5.5


def test_retry_after_is_capped_by_policy() -> None:
    transport, _seen = _sequence(
        httpx.Response(429, headers={"Retry-After": "3600"}), httpx.Response(200, content=b"ok")
    )
    sleeps: list[float] = []
    with httpx.Client(transport=transport) as client:
        post_with_retry(
            client, _URL, policy=RetryPolicy(max_retry_after_seconds=7.5), sleep=sleeps.append
        )
    assert sleeps == [7.5]


def test_persistent_5xx_exhausts_retries_with_backoff_schedule() -> None:
    transport, seen = _sequence(*[httpx.Response(503, content=b"down") for _ in range(6)])
    sleeps: list[float] = []
    with (
        httpx.Client(transport=transport) as client,
        pytest.raises(TTSError, match="503") as exc_info,
    ):
        post_with_retry(client, _URL, sleep=sleeps.append)
    assert seen[0] == 6
    assert sleeps == [1, 2, 4, 8, 16]
    assert isinstance(exc_info.value.__cause__, httpx.HTTPStatusError)


def test_connect_error_is_retried_then_succeeds() -> None:
    transport, seen = _sequence(httpx.ConnectError("refused"), httpx.Response(200, content=b"ok"))
    sleeps: list[float] = []
    with httpx.Client(transport=transport) as client:
        response = post_with_retry(client, _URL, sleep=sleeps.append)
    assert response.status_code == 200
    assert seen[0] == 2
    assert sleeps == [1]


def test_persistent_transport_error_chains_the_last_exception() -> None:
    transport, seen = _sequence(*[httpx.ReadTimeout("slow") for _ in range(3)])
    sleeps: list[float] = []
    with (
        httpx.Client(transport=transport) as client,
        pytest.raises(TTSError, match="ReadTimeout") as exc_info,
    ):
        post_with_retry(client, _URL, policy=RetryPolicy(max_retries=2), sleep=sleeps.append)
    assert seen[0] == 3
    assert sleeps == [1, 2]
    assert isinstance(exc_info.value.__cause__, httpx.ReadTimeout)


@pytest.mark.parametrize("status", [400, 401, 403, 404, 422])
def test_non_retryable_status_fails_immediately_with_body_excerpt(status: int) -> None:
    body = b"x" * 500
    transport, seen = _sequence(httpx.Response(status, content=body))
    sleeps: list[float] = []
    with (
        httpx.Client(transport=transport) as client,
        pytest.raises(TTSError, match=str(status)) as exc_info,
    ):
        post_with_retry(client, _URL, sleep=sleeps.append)
    assert seen[0] == 1
    assert sleeps == []
    assert "x" * 200 in str(exc_info.value)
    assert "x" * 201 not in str(exc_info.value)


def test_pcm_response_to_clip_decodes_s16le() -> None:
    pcm = np.array([0, 16384, -32768, 32767], dtype="<i2").tobytes()
    clip = pcm_response_to_clip(httpx.Response(200, content=pcm), 24000)
    assert clip.sample_rate == 24000
    assert clip.samples.dtype == np.float32
    np.testing.assert_allclose(clip.samples, [0.0, 0.5, -1.0, 32767 / 32768])


def test_pcm_response_to_clip_rejects_odd_length() -> None:
    with pytest.raises(TTSError, match="odd"):
        pcm_response_to_clip(httpx.Response(200, content=b"\x00\x01\x02"), 24000)
