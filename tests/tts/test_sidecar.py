from __future__ import annotations

import os
import subprocess
from pathlib import Path
from unittest.mock import MagicMock

import httpx
import pytest

from epub_to_m4b.tts import sidecar


def _ok(sample_rate: int = 24000) -> httpx.Response:
    return httpx.Response(200, json={"status": "ok", "sample_rate": sample_rate})


def _loading() -> httpx.Response:
    return httpx.Response(503, json={"status": "loading"})


def _health_transport(responses: list[httpx.Response]) -> httpx.MockTransport:
    """Return a MockTransport that yields ``responses`` in order, then repeats the last."""
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        index = min(calls["n"], len(responses) - 1)
        calls["n"] += 1
        return responses[index]

    return httpx.MockTransport(handler)


def _spawn_then_transport(responses: list[httpx.Response]) -> httpx.MockTransport:
    """A MockTransport for the "nothing listening yet" path.

    The very first health check raises a connection error (nothing bound to
    the port at all), which is the only outcome that should trigger a spawn.
    Every call after that steps through ``responses`` (repeating the last).
    """
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        n = calls["n"]
        calls["n"] += 1
        if n == 0:
            raise httpx.ConnectError("connection refused")
        index = min(n - 1, len(responses) - 1)
        return responses[index]

    return httpx.MockTransport(handler)


def test_adopts_already_healthy_server_without_spawning(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    popen = MagicMock()
    monkeypatch.setattr(subprocess, "Popen", popen)

    transport = _health_transport([_ok()])
    handle = sidecar.start_or_adopt(
        ["fake-command"],
        7861,
        log_path=tmp_path / "server.log",
        transport=transport,
    )

    assert handle.owned is False
    assert handle.sample_rate == 24000
    assert handle.base_url == "http://127.0.0.1:7861"
    popen.assert_not_called()


def test_spawns_when_nothing_listening_and_waits_through_loading(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Connection failure on the very first check - and only that - triggers a spawn."""
    fake_process = MagicMock(spec=subprocess.Popen)
    fake_process.poll.return_value = None
    popen = MagicMock(return_value=fake_process)
    monkeypatch.setattr(subprocess, "Popen", popen)
    monkeypatch.setattr(sidecar.time, "sleep", lambda _seconds: None)

    transport = _spawn_then_transport(
        [
            _loading(),
            _loading(),
            _ok(22050),
        ]
    )
    log_path = tmp_path / "sub" / "server.log"
    handle = sidecar.start_or_adopt(
        ["fake-command"],
        7861,
        log_path=log_path,
        transport=transport,
    )

    assert handle.owned is True
    assert handle.sample_rate == 22050
    assert handle.process is fake_process
    popen.assert_called_once()
    spawned_args = popen.call_args.args[0]
    assert spawned_args[:1] == ["fake-command"]
    assert spawned_args[-4:] == ["--host", "127.0.0.1", "--port", "7861"]
    assert log_path.parent.is_dir()


def test_adopts_already_running_server_still_loading_on_first_check(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A real (non-200) response on the first check means something is
    already bound to the port and mid-startup - it must be adopted, not
    competed with by spawning a second process on top of it.
    """
    popen = MagicMock()
    monkeypatch.setattr(subprocess, "Popen", popen)
    monkeypatch.setattr(sidecar.time, "sleep", lambda _seconds: None)

    transport = _health_transport([_loading(), _loading(), _ok(24000)])
    handle = sidecar.start_or_adopt(
        ["fake-command"], 7861, log_path=tmp_path / "server.log", transport=transport
    )

    assert handle.owned is False
    assert handle.process is None
    assert handle.sample_rate == 24000
    popen.assert_not_called()


def test_spawn_layers_caller_env_over_process_env(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("KEEP_ME", "from-parent")
    monkeypatch.setenv("OVERRIDE_ME", "from-parent")
    fake_process = MagicMock(spec=subprocess.Popen)
    fake_process.poll.return_value = None
    popen = MagicMock(return_value=fake_process)
    monkeypatch.setattr(subprocess, "Popen", popen)

    transport = _spawn_then_transport([_ok()])
    sidecar.start_or_adopt(
        ["fake-command"],
        7861,
        log_path=tmp_path / "server.log",
        env={"OVERRIDE_ME": "from-caller", "ONLY_CALLER": "x"},
        transport=transport,
    )

    env = popen.call_args.kwargs["env"]
    assert env["KEEP_ME"] == "from-parent"
    assert env["OVERRIDE_ME"] == "from-caller"
    assert env["ONLY_CALLER"] == "x"


def test_spawn_without_env_passes_process_env_unchanged(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """sidecar.py is engine-agnostic: no engine-specific variables (e.g.
    TRITON_PTXAS_PATH) are injected here - that is the engine's job.
    """
    monkeypatch.delenv("TRITON_PTXAS_PATH", raising=False)
    fake_process = MagicMock(spec=subprocess.Popen)
    fake_process.poll.return_value = None
    popen = MagicMock(return_value=fake_process)
    monkeypatch.setattr(subprocess, "Popen", popen)

    transport = _spawn_then_transport([_ok()])
    sidecar.start_or_adopt(
        ["fake-command"], 7861, log_path=tmp_path / "server.log", transport=transport
    )

    env = popen.call_args.kwargs["env"]
    assert "TRITON_PTXAS_PATH" not in env
    assert env == dict(os.environ)


def test_timeout_raises_with_log_path_in_message(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake_process = MagicMock(spec=subprocess.Popen)
    fake_process.poll.return_value = None
    popen = MagicMock(return_value=fake_process)
    monkeypatch.setattr(subprocess, "Popen", popen)
    monkeypatch.setattr(sidecar.time, "sleep", lambda _seconds: None)

    # Nothing listening initially (triggers our spawn), then always loading -
    # never becomes healthy.
    transport = _spawn_then_transport([_loading()])
    log_path = tmp_path / "server.log"

    with pytest.raises(sidecar.SidecarError) as exc_info:
        sidecar.start_or_adopt(
            ["fake-command"],
            7861,
            log_path=log_path,
            policy=sidecar.SidecarPolicy(startup_timeout=0.05, poll_interval=0.01),
            transport=transport,
        )

    assert str(log_path) in str(exc_info.value)
    fake_process.terminate.assert_called_once()


def test_timeout_keeps_spawned_process_terminated_not_adopted(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake_process = MagicMock(spec=subprocess.Popen)
    fake_process.poll.return_value = None
    popen = MagicMock(return_value=fake_process)
    monkeypatch.setattr(subprocess, "Popen", popen)
    monkeypatch.setattr(sidecar.time, "sleep", lambda _seconds: None)

    transport = _spawn_then_transport([_loading()])
    with pytest.raises(sidecar.SidecarError):
        sidecar.start_or_adopt(
            ["fake-command"],
            7861,
            log_path=tmp_path / "server.log",
            policy=sidecar.SidecarPolicy(startup_timeout=0.02, poll_interval=0.01),
            transport=transport,
        )
    fake_process.terminate.assert_called()


def test_timeout_on_adopted_never_healthy_server_terminates_nothing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """If we never spawned anything (something else was already there but
    never becomes healthy), timing out must not reach for a process to kill.
    """
    popen = MagicMock()
    monkeypatch.setattr(subprocess, "Popen", popen)
    monkeypatch.setattr(sidecar.time, "sleep", lambda _seconds: None)

    transport = _health_transport([_loading()])
    with pytest.raises(sidecar.SidecarError):
        sidecar.start_or_adopt(
            ["fake-command"],
            7861,
            log_path=tmp_path / "server.log",
            policy=sidecar.SidecarPolicy(startup_timeout=0.02, poll_interval=0.01),
            transport=transport,
        )
    popen.assert_not_called()


def test_connection_refused_is_treated_as_not_running_and_spawns(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake_process = MagicMock(spec=subprocess.Popen)
    fake_process.poll.return_value = None
    popen = MagicMock(return_value=fake_process)
    monkeypatch.setattr(subprocess, "Popen", popen)

    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            raise httpx.ConnectError("connection refused")
        return _ok()

    transport = httpx.MockTransport(handler)
    handle = sidecar.start_or_adopt(
        ["fake-command"], 7861, log_path=tmp_path / "server.log", transport=transport
    )

    assert handle.owned is True
    popen.assert_called_once()


def test_close_terminates_owned_process() -> None:
    fake_process = MagicMock(spec=subprocess.Popen)
    fake_process.poll.return_value = None
    handle = sidecar.SidecarHandle(
        base_url="http://127.0.0.1:7861",
        sample_rate=24000,
        owned=True,
        process=fake_process,
        log_path=Path("/tmp/does-not-matter.log"),
    )

    handle.close()

    fake_process.terminate.assert_called_once()
    fake_process.wait.assert_called_once()


def test_close_does_not_terminate_adopted_process() -> None:
    fake_process = MagicMock(spec=subprocess.Popen)
    handle = sidecar.SidecarHandle(
        base_url="http://127.0.0.1:7861",
        sample_rate=24000,
        owned=False,
        process=fake_process,
        log_path=Path("/tmp/does-not-matter.log"),
    )

    handle.close()

    fake_process.terminate.assert_not_called()


def test_close_is_a_no_op_if_process_already_exited() -> None:
    fake_process = MagicMock(spec=subprocess.Popen)
    fake_process.poll.return_value = 0
    handle = sidecar.SidecarHandle(
        base_url="http://127.0.0.1:7861",
        sample_rate=24000,
        owned=True,
        process=fake_process,
        log_path=Path("/tmp/does-not-matter.log"),
    )

    handle.close()

    fake_process.terminate.assert_not_called()


def test_close_kills_if_terminate_does_not_exit_in_time() -> None:
    fake_process = MagicMock(spec=subprocess.Popen)
    fake_process.poll.return_value = None
    fake_process.wait.side_effect = [subprocess.TimeoutExpired(cmd="x", timeout=1), None]
    handle = sidecar.SidecarHandle(
        base_url="http://127.0.0.1:7861",
        sample_rate=24000,
        owned=True,
        process=fake_process,
        log_path=Path("/tmp/does-not-matter.log"),
    )

    handle.close()

    fake_process.terminate.assert_called_once()
    fake_process.kill.assert_called_once()


def test_writes_subprocess_output_to_log_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake_process = MagicMock(spec=subprocess.Popen)
    fake_process.poll.return_value = None
    popen = MagicMock(return_value=fake_process)
    monkeypatch.setattr(subprocess, "Popen", popen)

    transport = _health_transport([_ok()])
    log_path = tmp_path / "server.log"
    sidecar.start_or_adopt(["fake-command"], 7861, log_path=log_path, transport=transport)

    # health was already OK, so we never spawn - but confirm the parent dir
    # creation/log wiring path is exercised in the spawn tests above, and
    # here that no log file is created when we adopt instead of spawn.
    popen.assert_not_called()
    assert not log_path.exists()


def test_spawn_writes_stdout_stderr_to_log_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    captured: dict[str, object] = {}

    def fake_popen(args: list[str], **kwargs: object) -> MagicMock:
        captured.update(kwargs)
        process = MagicMock()
        process.poll.return_value = None
        return process

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    transport = _spawn_then_transport([_ok()])
    log_path = tmp_path / "server.log"

    sidecar.start_or_adopt(["fake-command"], 7861, log_path=log_path, transport=transport)

    assert captured["stderr"] == subprocess.STDOUT
    assert log_path.exists()


def _busy() -> httpx.Response:
    return httpx.Response(409, json={"detail": "busy"})


def test_post_until_free_retries_busy_then_returns_success() -> None:
    responses = iter([_busy(), _busy(), httpx.Response(200, content=b"pcm")])
    sleeps: list[float] = []
    on_busy = MagicMock()
    policy = sidecar.SidecarPolicy(busy_timeout=1.25, busy_wait=0.25)

    response = sidecar.post_until_free(
        lambda: next(responses), policy=policy, on_busy=on_busy, sleep=sleeps.append
    )

    assert response.status_code == 200
    assert response.content == b"pcm"
    on_busy.assert_called_once()
    assert sleeps == [0.25, 0.25]


def test_post_until_free_exhausted_returns_last_busy_response() -> None:
    calls = {"n": 0}

    def send() -> httpx.Response:
        calls["n"] += 1
        return _busy()

    sleeps: list[float] = []
    policy = sidecar.SidecarPolicy(busy_timeout=1.0, busy_wait=0.5)

    response = sidecar.post_until_free(send, policy=policy, sleep=sleeps.append)

    assert response.status_code == 409
    assert calls["n"] == 3
    assert sleeps == [0.5, 0.5]


def test_post_until_free_returns_non_busy_immediately() -> None:
    calls = {"n": 0}

    def send() -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(500)

    on_busy = MagicMock()
    sleep = MagicMock()

    response = sidecar.post_until_free(
        send, policy=sidecar.SidecarPolicy(), on_busy=on_busy, sleep=sleep
    )

    assert response.status_code == 500
    assert calls["n"] == 1
    on_busy.assert_not_called()
    sleep.assert_not_called()


def test_post_until_free_honours_custom_busy_status() -> None:
    responses = iter([httpx.Response(503), httpx.Response(200)])
    policy = sidecar.SidecarPolicy(busy_status=503, busy_timeout=1.0, busy_wait=1.0)

    response = sidecar.post_until_free(lambda: next(responses), policy=policy, sleep=lambda _: None)

    assert response.status_code == 200


def test_busy_retries_derive_from_timeout_and_wait() -> None:
    assert sidecar.SidecarPolicy(busy_timeout=300.0, busy_wait=5.0).busy_retries == 60
    # A partial final interval still gets its retry, so the wait covers the timeout.
    assert sidecar.SidecarPolicy(busy_timeout=10.0, busy_wait=3.0).busy_retries == 4
    assert sidecar.SidecarPolicy(busy_timeout=0.0).busy_retries == 0


def test_busy_wait_must_be_positive() -> None:
    with pytest.raises(ValueError, match="busy_wait"):
        sidecar.SidecarPolicy(busy_wait=0.0)
