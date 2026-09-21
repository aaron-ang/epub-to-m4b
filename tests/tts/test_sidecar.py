from __future__ import annotations

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


def test_spawns_and_waits_through_loading(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    fake_process = MagicMock(spec=subprocess.Popen)
    fake_process.poll.return_value = None
    popen = MagicMock(return_value=fake_process)
    monkeypatch.setattr(subprocess, "Popen", popen)
    monkeypatch.setattr(sidecar.time, "sleep", lambda _seconds: None)

    transport = _health_transport(
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


def test_spawn_sets_default_triton_ptxas_path_when_unset(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("TRITON_PTXAS_PATH", raising=False)
    fake_process = MagicMock(spec=subprocess.Popen)
    fake_process.poll.return_value = None
    popen = MagicMock(return_value=fake_process)
    monkeypatch.setattr(subprocess, "Popen", popen)

    transport = _health_transport(
        [
            _loading(),
            _ok(),
        ]
    )
    sidecar.start_or_adopt(
        ["fake-command"], 7861, log_path=tmp_path / "server.log", transport=transport
    )

    env = popen.call_args.kwargs["env"]
    assert env["TRITON_PTXAS_PATH"] == "/usr/local/cuda/bin/ptxas"


def test_spawn_respects_already_set_triton_ptxas_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("TRITON_PTXAS_PATH", "/custom/ptxas")
    fake_process = MagicMock(spec=subprocess.Popen)
    fake_process.poll.return_value = None
    popen = MagicMock(return_value=fake_process)
    monkeypatch.setattr(subprocess, "Popen", popen)

    transport = _health_transport(
        [
            _loading(),
            _ok(),
        ]
    )
    sidecar.start_or_adopt(
        ["fake-command"], 7861, log_path=tmp_path / "server.log", transport=transport
    )

    env = popen.call_args.kwargs["env"]
    assert env["TRITON_PTXAS_PATH"] == "/custom/ptxas"


def test_timeout_raises_with_log_path_in_message(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake_process = MagicMock(spec=subprocess.Popen)
    fake_process.poll.return_value = None
    popen = MagicMock(return_value=fake_process)
    monkeypatch.setattr(subprocess, "Popen", popen)
    monkeypatch.setattr(sidecar.time, "sleep", lambda _seconds: None)

    # Always loading - never becomes healthy.
    transport = _health_transport([_loading()])
    log_path = tmp_path / "server.log"

    with pytest.raises(TimeoutError) as exc_info:
        sidecar.start_or_adopt(
            ["fake-command"],
            7861,
            log_path=log_path,
            startup_timeout=0.05,
            poll_interval=0.01,
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

    transport = _health_transport([_loading()])
    with pytest.raises(TimeoutError):
        sidecar.start_or_adopt(
            ["fake-command"],
            7861,
            log_path=tmp_path / "server.log",
            startup_timeout=0.02,
            poll_interval=0.01,
            transport=transport,
        )
    fake_process.terminate.assert_called()


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
    transport = _health_transport(
        [
            _loading(),
            _ok(),
        ]
    )
    log_path = tmp_path / "server.log"

    sidecar.start_or_adopt(["fake-command"], 7861, log_path=log_path, transport=transport)

    assert captured["stderr"] == subprocess.STDOUT
    assert log_path.exists()
