"""Retrying POST helper shared by the HTTP API engines.

Every hosted TTS API fails the same few ways - a dropped connection, a
timeout, a 429 when the account's rate limit trips, a 5xx while the
provider is degraded - and recovers the same way: wait, then send the same
request again. ``post_with_retry`` owns that loop so each engine module is
only the request shape and the response decoding. Rate limiting is left
entirely to the server: a 429 with ``Retry-After`` tells us exactly how
long to wait, so there is no client-side token bucket to keep in step with
each provider's quota rules.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import Any

import httpx
import numpy as np

from epub_to_m4b.book import AudioClip
from epub_to_m4b.errors import EpubToM4bError
from epub_to_m4b.tts.base import pcm16_to_float32

_RETRYABLE_STATUSES = frozenset({httpx.codes.REQUEST_TIMEOUT, httpx.codes.TOO_MANY_REQUESTS})
_BODY_EXCERPT_CHARS = 200
# Speech for one sentence takes seconds; connecting should not.
_TIMEOUT = httpx.Timeout(120.0, connect=10.0)


class TTSError(EpubToM4bError):
    """A TTS request failed for good: retries exhausted, a status the
    server will not recover from, or a malformed response body."""


@dataclass(frozen=True)
class RetryPolicy:
    # Retry n waits backoff_base_seconds * 2**n: doubling waits ride out a
    # transient outage without hammering the provider.
    max_retries: int = 5
    backoff_base_seconds: float = 1.0
    # Cap on an honoured Retry-After header so a huge value from the
    # provider does not stall the run indefinitely.
    max_retry_after_seconds: float = 60.0


DEFAULT_RETRY_POLICY = RetryPolicy()


def new_client(transport: httpx.BaseTransport | None = None) -> httpx.Client:
    """An ``httpx.Client`` with the shared API timeouts; ``transport`` lets
    tests substitute a ``MockTransport`` for the network."""
    if transport is None:
        return httpx.Client(timeout=_TIMEOUT)
    return httpx.Client(timeout=_TIMEOUT, transport=transport)


def _is_retryable(status: int) -> bool:
    return status in _RETRYABLE_STATUSES or status >= httpx.codes.INTERNAL_SERVER_ERROR


def _retry_after_seconds(response: httpx.Response) -> float | None:
    """The server's own wait hint, as either integer seconds or an
    HTTP-date; ``None`` when absent or unparseable."""
    header = response.headers.get("Retry-After")
    if header is None:
        return None
    header = header.strip()
    if header.isdigit():
        return float(header)
    try:
        when = parsedate_to_datetime(header)
    except TypeError, ValueError:
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=UTC)
    return max(0.0, (when - datetime.now(UTC)).total_seconds())


def _wait_before_retry(response: httpx.Response | None, attempt: int, policy: RetryPolicy) -> float:
    if response is not None:
        hinted = _retry_after_seconds(response)
        if hinted is not None:
            return min(hinted, policy.max_retry_after_seconds)
    return policy.backoff_base_seconds * 2.0**attempt


def _excerpt(response: httpx.Response) -> str:
    return response.text[:_BODY_EXCERPT_CHARS]


def post_with_retry(
    client: httpx.Client,
    url: str,
    *,
    policy: RetryPolicy = DEFAULT_RETRY_POLICY,
    sleep: Callable[[float], None] = time.sleep,
    **request_kwargs: Any,
) -> httpx.Response:
    """POST ``url`` until it returns 2xx, retrying transport errors and
    408/429/5xx with ``Retry-After`` or the policy's backoff schedule in
    between. Any other non-2xx status is final and raises at once."""
    last_error: Exception | None = None
    last_response: httpx.Response | None = None
    for attempt in range(policy.max_retries + 1):
        last_response = None
        try:
            response = client.post(url, **request_kwargs)
        except httpx.TransportError as exc:
            last_error = exc
        else:
            if response.is_success:
                return response
            if not _is_retryable(response.status_code):
                raise TTSError(
                    f"POST {url} failed with status {response.status_code}: {_excerpt(response)}"
                )
            last_response = response
            last_error = httpx.HTTPStatusError(
                f"status {response.status_code}", request=response.request, response=response
            )
        if attempt < policy.max_retries:
            sleep(_wait_before_retry(last_response, attempt, policy))

    assert last_error is not None  # the loop always runs at least once
    if last_response is not None:
        detail = f"status {last_response.status_code}: {_excerpt(last_response)}"
    else:
        detail = f"{type(last_error).__name__}: {last_error}"
    raise TTSError(
        f"POST {url} failed after {policy.max_retries + 1} attempts, last {detail}"
    ) from last_error


def pcm_response_to_clip(response: httpx.Response, sample_rate: int) -> AudioClip:
    """Decode a raw s16le mono body into a float32 ``AudioClip``."""
    body = response.content
    if len(body) % 2:
        raise TTSError(f"PCM response body has odd length {len(body)}; expected s16le samples")
    pcm = np.frombuffer(body, dtype="<i2")
    return AudioClip(samples=pcm16_to_float32(pcm), sample_rate=sample_rate)
