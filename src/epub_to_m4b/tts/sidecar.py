"""Spawn or adopt a local HTTP TTS server.

Self-hosted engines (Breeze today, others later) run as a separate process
so this project never needs to import their heavy ML dependencies directly.
This module is deliberately engine-agnostic: it knows nothing about Breeze's
request/response shapes, only that ``GET {base_url}/health`` answers ``200
{"status": "ok", "sample_rate": N}`` once ready and ``503
{"status": "loading"}`` while starting up.

If a server is already listening on the port, it's adopted rather than
replaced - useful when a long-lived sidecar outlives one CLI invocation, or
when something else is already using it. An adopted server is never killed
by ``close()``.
"""

from __future__ import annotations

import os
import subprocess
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Self

import httpx

_HEALTH_TIMEOUT_SECONDS = 2.0
_POLL_INTERVAL_SECONDS = 2.0
_STARTUP_TIMEOUT_SECONDS = 180.0
_TERMINATE_TIMEOUT_SECONDS = 10.0
# Documented real gotcha: without this, triton can't find ptxas in this
# environment and the server fails to start.
_TRITON_PTXAS_PATH_ENV = "TRITON_PTXAS_PATH"
_DEFAULT_TRITON_PTXAS_PATH = "/usr/local/cuda/bin/ptxas"


def _poll_health(client: httpx.Client, base_url: str) -> httpx.Response | None:
    """Return the /health response, or None if the connection itself failed.

    A connection failure (refused/timeout) means "not up yet" and should
    keep the caller waiting or spawning; any actual HTTP response - even a
    500 - means something is listening and its status code/body decide
    what happens next.
    """
    try:
        return client.get(f"{base_url}/health", timeout=_HEALTH_TIMEOUT_SECONDS)
    except httpx.TransportError:
        return None


def _terminate(
    process: subprocess.Popen[bytes], timeout: float = _TERMINATE_TIMEOUT_SECONDS
) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=timeout)


@dataclass
class SidecarHandle:
    """A local HTTP server we either spawned or adopted."""

    base_url: str
    sample_rate: int
    owned: bool
    process: subprocess.Popen[bytes] | None
    log_path: Path

    def close(self) -> None:
        """Terminate the server if we spawned it; leave an adopted one running."""
        if self.owned and self.process is not None:
            _terminate(self.process)

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
    startup_timeout: float = _STARTUP_TIMEOUT_SECONDS,
    poll_interval: float = _POLL_INTERVAL_SECONDS,
    transport: httpx.BaseTransport | None = None,
) -> SidecarHandle:
    """Adopt an already-healthy server on ``port``, or spawn ``command`` there.

    ``command`` already includes any engine-specific args (e.g. the weights
    directory); this function appends ``--host 127.0.0.1 --port {port}``.
    Blocks until the server answers healthy or ``startup_timeout`` elapses.
    ``transport`` overrides the HTTP transport used for the health checks
    (for tests; real callers leave it as the default).
    """
    base_url = f"http://127.0.0.1:{port}"
    client_kwargs: dict[str, httpx.BaseTransport] = {}
    if transport is not None:
        client_kwargs["transport"] = transport

    with httpx.Client(**client_kwargs) as client:  # type: ignore[arg-type]
        response = _poll_health(client, base_url)
        if response is not None and response.status_code == 200:
            sample_rate = int(response.json()["sample_rate"])
            return SidecarHandle(
                base_url=base_url,
                sample_rate=sample_rate,
                owned=False,
                process=None,
                log_path=log_path,
            )

        process = _spawn(command, port, log_path)
        try:
            deadline = time.monotonic() + startup_timeout
            while time.monotonic() < deadline:
                response = _poll_health(client, base_url)
                if response is not None and response.status_code == 200:
                    sample_rate = int(response.json()["sample_rate"])
                    return SidecarHandle(
                        base_url=base_url,
                        sample_rate=sample_rate,
                        owned=True,
                        process=process,
                        log_path=log_path,
                    )
                # A non-200 response (e.g. 503 loading) or connection failure
                # both mean "keep waiting" here - only a 200 ends the wait.
                time.sleep(poll_interval)

            raise TimeoutError(
                f"server on port {port} did not become healthy within "
                f"{startup_timeout:.0f}s; see log at {log_path}"
            )
        except BaseException:
            _terminate(process)
            raise


def _spawn(command: Sequence[str], port: int, log_path: Path) -> subprocess.Popen[bytes]:
    env = dict(os.environ)
    env.setdefault(_TRITON_PTXAS_PATH_ENV, _DEFAULT_TRITON_PTXAS_PATH)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    full_command = [*command, "--host", "127.0.0.1", "--port", str(port)]
    with log_path.open("wb") as log_file:
        return subprocess.Popen(
            full_command,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            env=env,
        )
