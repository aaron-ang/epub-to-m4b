"""Lifecycle and busy-wait layer for any single-slot local inference server.

Self-hosted engines (e.g. Breeze) run as a separate HTTP process so this
project never needs to import their heavy ML dependencies directly. This
module is deliberately engine-agnostic: it knows nothing about any engine's
request/response shapes, only that

* ``GET {base_url}/health`` answers ``200 {"status": "ok", "sample_rate": N}``
  once ready and ``503 {"status": "loading"}`` while starting up, and
* the server serves one inference at a time and answers a fixed "busy"
  status (409 by default) while another request is running.

Two pieces live here:

* :func:`start_or_adopt` - spawn the server or adopt one already listening
  on the port, then block until it reports healthy. An adopted server is
  never killed by ``close()``.
* :func:`post_until_free` - retry a request while the server answers busy.

All timing and retry knobs are grouped in :class:`SidecarPolicy` so every
engine shares one vocabulary and tests can shrink the waits.
"""

from __future__ import annotations

import math
import os
import subprocess
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Self

import httpx

from epub_to_m4b.errors import EpubToM4bError


class SidecarError(EpubToM4bError):
    """The local TTS server could not be started or never reported healthy."""


@dataclass(frozen=True)
class SidecarPolicy:
    """Timing and busy-wait rules for any single-slot local inference server."""

    health_timeout: float = 2.0  # one /health request
    poll_interval: float = 2.0  # spacing between /health polls while starting
    startup_timeout: float = 180.0  # model load on a cold GPU takes minutes
    terminate_timeout: float = 10.0  # graceful exit before SIGKILL
    request_timeout: float = 300.0  # single non-batched request (bootstrap voice etc.)
    batch_timeout: float = 1800.0  # a full batch decodes for minutes, not seconds
    busy_status: int = httpx.codes.CONFLICT  # server answers this while another inference runs
    busy_timeout: float = 300.0  # total wait for a busy server before giving up
    busy_wait: float = 5.0  # between busy retries

    def __post_init__(self) -> None:
        if self.busy_wait <= 0:
            raise ValueError(f"busy_wait must be positive, got {self.busy_wait}")

    @property
    def busy_retries(self) -> int:
        """Retries that fit in ``busy_timeout`` at ``busy_wait`` spacing."""
        return math.ceil(self.busy_timeout / self.busy_wait)


DEFAULT_SIDECAR_POLICY = SidecarPolicy()


def _poll_health(
    client: httpx.Client, base_url: str, policy: SidecarPolicy
) -> httpx.Response | None:
    """Return the /health response, or None if the connection itself failed.

    A connection failure (refused/timeout) means "not up yet" and should
    keep the caller waiting or spawning; any actual HTTP response - even a
    500 - means something is listening and its status code/body decide
    what happens next.
    """
    try:
        return client.get(f"{base_url}/health", timeout=policy.health_timeout)
    except httpx.TransportError:
        return None


def _terminate(process: subprocess.Popen[bytes], policy: SidecarPolicy) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=policy.terminate_timeout)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=policy.terminate_timeout)


@dataclass
class SidecarHandle:
    """A local HTTP server we either spawned or adopted."""

    base_url: str
    sample_rate: int
    owned: bool
    process: subprocess.Popen[bytes] | None
    log_path: Path
    policy: SidecarPolicy = DEFAULT_SIDECAR_POLICY

    def close(self) -> None:
        """Terminate the server if we spawned it; leave an adopted one running."""
        if self.owned and self.process is not None:
            _terminate(self.process, self.policy)

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()


def start_or_adopt(
    command: Sequence[str],
    port: int,
    *,
    log_path: Path,
    policy: SidecarPolicy = DEFAULT_SIDECAR_POLICY,
    env: Mapping[str, str] | None = None,
    transport: httpx.BaseTransport | None = None,
) -> SidecarHandle:
    """Adopt an already-running server on ``port``, or spawn ``command`` there.

    The initial check distinguishes three outcomes, not two:

    1. Connection failure (nothing listening at all) - we spawn ``command``.
    2. A real HTTP response that isn't 200 (e.g. 503 loading) - something is
       already bound to this port and mid-startup. We must not spawn a
       competing process on top of it; adopt it (``owned=False``) and keep
       polling the same way we would after our own spawn.
    3. A real 200 response - adopt immediately, no polling needed.

    ``command`` already includes any engine-specific args (e.g. the model
    path); this function appends ``--host 127.0.0.1 --port {port}``.
    ``env`` is layered over the current process environment for the spawned
    child (caller's entries win); engine-specific variables belong to the
    caller, not here. Blocks until the server answers healthy or
    ``policy.startup_timeout`` elapses. ``transport`` overrides the HTTP
    transport used for the health checks (for tests; real callers leave it
    as the default).
    """
    base_url = f"http://127.0.0.1:{port}"
    client_kwargs: dict[str, httpx.BaseTransport] = {}
    if transport is not None:
        client_kwargs["transport"] = transport

    with httpx.Client(**client_kwargs) as client:  # type: ignore[arg-type]
        initial = _poll_health(client, base_url, policy)
        if initial is not None and initial.status_code == httpx.codes.OK:
            return SidecarHandle(
                base_url=base_url,
                sample_rate=int(initial.json()["sample_rate"]),
                owned=False,
                process=None,
                log_path=log_path,
                policy=policy,
            )

        process: subprocess.Popen[bytes] | None = None
        owned = False
        if initial is None:
            # Nothing answered at all - the port is genuinely free, spawn ours.
            process = _spawn(command, port, log_path, env)
            owned = True
        # else: a real, non-200 response (e.g. 503 loading) - something else
        # already owns this port and is mid-startup; fall through to the
        # same polling loop without spawning, and adopt it once healthy.

        try:
            deadline = time.monotonic() + policy.startup_timeout
            while time.monotonic() < deadline:
                response = _poll_health(client, base_url, policy)
                if response is not None and response.status_code == httpx.codes.OK:
                    return SidecarHandle(
                        base_url=base_url,
                        sample_rate=int(response.json()["sample_rate"]),
                        owned=owned,
                        process=process,
                        log_path=log_path,
                        policy=policy,
                    )
                # A non-200 response (e.g. 503 loading) or connection failure
                # both mean "keep waiting" here - only a 200 ends the wait.
                time.sleep(policy.poll_interval)

            raise SidecarError(
                f"server on port {port} did not become healthy within "
                f"{policy.startup_timeout:.0f}s; see log at {log_path}"
            )
        except BaseException:
            if process is not None:
                _terminate(process, policy)
            raise


def _spawn(
    command: Sequence[str], port: int, log_path: Path, env: Mapping[str, str] | None
) -> subprocess.Popen[bytes]:
    child_env = {**os.environ, **(env or {})}
    log_path.parent.mkdir(parents=True, exist_ok=True)
    full_command = [*command, "--host", "127.0.0.1", "--port", str(port)]
    with log_path.open("wb") as log_file:
        return subprocess.Popen(
            full_command,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            env=child_env,
        )


def post_until_free(
    send: Callable[[], httpx.Response],
    *,
    policy: SidecarPolicy,
    on_busy: Callable[[], None] | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> httpx.Response:
    """Call ``send()`` until the server stops answering ``policy.busy_status``.

    Tries at most ``policy.busy_retries + 1`` times, sleeping
    ``policy.busy_wait`` between attempts, so the total wait is about
    ``policy.busy_timeout``. ``on_busy`` runs once, on the
    first busy answer, so callers can log without spamming. Returns the last
    response - possibly still busy - and never raises on status; the caller
    decides with ``raise_for_status()``. ``send`` is zero-arg so callers can
    reopen file handles per attempt.
    """
    warned = False
    response = send()
    for _ in range(policy.busy_retries):
        if response.status_code != policy.busy_status:
            break
        if not warned and on_busy is not None:
            on_busy()
        warned = True
        sleep(policy.busy_wait)
        response = send()
    return response
