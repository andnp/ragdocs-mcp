from __future__ import annotations

import subprocess
import signal
import sys
from typing import Any, cast
from pathlib import Path

from mcp_markdown_ragdocs.daemon.paths import RuntimePaths
from mcp_markdown_ragdocs.worker.process import (
    HueyWorkerProcess,
    _read_worker_status,
    _remove_worker_status,
    is_expected_daemon_parent,
)


class _FakeProcess:
    def __init__(self):
        self.pid = 1234
        self._returncode = None
        self.signals: list[int] = []
        self.kill_calls = 0
        self.wait_calls: list[float] = []

    def poll(self):
        return self._returncode

    def send_signal(self, sig: int):
        self.signals.append(sig)
        self._returncode = 0

    def wait(self, timeout: float):
        self.wait_calls.append(timeout)
        if self._returncode is None:
            self._returncode = 0
        return self._returncode

    def kill(self):
        self.kill_calls += 1
        self._returncode = -9


class _HungProcess(_FakeProcess):
    def send_signal(self, sig: int):
        self.signals.append(sig)

    def wait(self, timeout: float):
        self.wait_calls.append(timeout)
        if self._returncode is not None:
            return self._returncode
        raise subprocess.TimeoutExpired("worker", timeout)


def _paths(tmp_path: Path) -> RuntimePaths:
    return RuntimePaths(
        root=tmp_path,
        index_db_path=tmp_path / "index.db",
        queue_db_path=tmp_path / "queue.db",
        metadata_path=tmp_path / "daemon.json",
        lock_path=tmp_path / "daemon.lock",
        socket_path=tmp_path / "daemon.sock",
    )


def test_worker_process_start_uses_internal_worker_command(monkeypatch, tmp_path: Path):
    observed: dict[str, Any] = {}
    fake_process = _FakeProcess()

    monkeypatch.setattr(
        "mcp_markdown_ragdocs.worker.process._resolve_daemon_python",
        lambda: Path("/repo/.venv/bin/python"),
    )
    monkeypatch.setattr(
        "mcp_markdown_ragdocs.worker.process.current_process_start_time_ticks",
        lambda: 424242,
    )
    monkeypatch.setattr(
        "mcp_markdown_ragdocs.worker.process._terminate_runtime_worker_processes",
        lambda _runtime_paths: None,
    )

    def _fake_popen(command: list[str], **kwargs: Any):
        observed["command"] = command
        observed["kwargs"] = kwargs
        return fake_process

    monkeypatch.setattr("mcp_markdown_ragdocs.worker.process.subprocess.Popen", _fake_popen)
    monkeypatch.setattr("mcp_markdown_ragdocs.worker.process.WORKER_STARTUP_TIMEOUT_SECONDS", 0.0)

    worker = HueyWorkerProcess(runtime_paths=_paths(tmp_path))
    worker.start()

    command = observed["command"]
    assert command[:4] == [
        "/repo/.venv/bin/python",
        "-m",
        "mcp_markdown_ragdocs.cli",
        "worker-run",
    ]
    assert "--queue-db" in command
    assert "--index-root" in command
    assert "--parent-pid" in command
    assert "--parent-start-time" in command
    assert "--project" not in command
    assert observed["kwargs"]["start_new_session"] is True
    assert worker.is_running is True


def test_worker_process_stop_sends_sigterm(monkeypatch, tmp_path: Path):
    fake_process = _FakeProcess()

    monkeypatch.setattr(
        "mcp_markdown_ragdocs.worker.process._resolve_daemon_python",
        lambda: Path("/repo/.venv/bin/python"),
    )
    monkeypatch.setattr(
        "mcp_markdown_ragdocs.worker.process._terminate_runtime_worker_processes",
        lambda _runtime_paths: None,
    )
    monkeypatch.setattr(
        "mcp_markdown_ragdocs.worker.process.subprocess.Popen",
        lambda *args, **kwargs: fake_process,
    )
    monkeypatch.setattr("mcp_markdown_ragdocs.worker.process.WORKER_STARTUP_TIMEOUT_SECONDS", 0.0)

    worker = HueyWorkerProcess(runtime_paths=_paths(tmp_path))
    worker.start()
    worker.stop(timeout=2.0)

    assert fake_process.signals == [signal.SIGTERM]
    assert worker.is_running is False


def test_worker_process_stop_kills_when_sigterm_times_out(
    monkeypatch,
    tmp_path: Path,
):
    fake_process = _HungProcess()
    monkeypatch.setattr(
        "mcp_markdown_ragdocs.worker.process._read_process_start_time_ticks",
        lambda _pid: 4242,
    )
    worker = HueyWorkerProcess(runtime_paths=_paths(tmp_path))
    worker._process = cast(Any, fake_process)

    worker.stop(timeout=0.0)

    assert fake_process.signals == [signal.SIGTERM]
    assert fake_process.kill_calls == 1
    assert worker.is_running is False


def test_worker_process_allows_long_running_task_heartbeat(
    monkeypatch,
    tmp_path: Path,
):
    fake_process = _FakeProcess()
    worker = HueyWorkerProcess(runtime_paths=_paths(tmp_path))
    worker._process = cast(Any, fake_process)

    monkeypatch.setattr(
        "mcp_markdown_ragdocs.worker.process._read_worker_status",
        lambda _runtime_paths: {
            "status": "ready",
            "pid": fake_process.pid,
            "heartbeat": 100.0,
        },
    )
    monkeypatch.setattr(
        "mcp_markdown_ragdocs.worker.process.time.time",
        lambda: 100.0 + 299.0,
    )

    assert worker.is_healthy() is True


def test_worker_process_restart_replaces_process(monkeypatch, tmp_path: Path):
    first = _FakeProcess()
    second = _FakeProcess()
    created = iter([first, second])

    monkeypatch.setattr(
        "mcp_markdown_ragdocs.worker.process._resolve_daemon_python",
        lambda: Path("/repo/.venv/bin/python"),
    )
    monkeypatch.setattr(
        "mcp_markdown_ragdocs.worker.process._terminate_runtime_worker_processes",
        lambda _runtime_paths: None,
    )
    monkeypatch.setattr(
        "mcp_markdown_ragdocs.worker.process.subprocess.Popen",
        lambda *args, **kwargs: next(created),
    )
    monkeypatch.setattr("mcp_markdown_ragdocs.worker.process.time.sleep", lambda _: None)
    monkeypatch.setattr("mcp_markdown_ragdocs.worker.process.WORKER_STARTUP_TIMEOUT_SECONDS", 0.0)

    worker = HueyWorkerProcess(runtime_paths=_paths(tmp_path))
    worker.start()
    assert worker.pid == first.pid

    worker.restart(timeout=2.0)

    assert first.signals == [signal.SIGTERM]
    assert worker.pid == second.pid


def test_is_expected_daemon_parent_requires_daemon_command(monkeypatch):
    monkeypatch.setattr("mcp_markdown_ragdocs.worker.process._process_exists", lambda _pid: True)
    monkeypatch.setattr(
        "mcp_markdown_ragdocs.worker.process._read_process_cmdline",
        lambda _pid: ["python", "-m", "mcp_markdown_ragdocs.cli", "query"],
    )

    assert is_expected_daemon_parent(1234, None) is False


def test_is_expected_daemon_parent_rejects_pid_reuse(monkeypatch):
    monkeypatch.setattr("mcp_markdown_ragdocs.worker.process._process_exists", lambda _pid: True)
    monkeypatch.setattr(
        "mcp_markdown_ragdocs.worker.process._read_process_cmdline",
        lambda _pid: ["python", "-m", "mcp_markdown_ragdocs.cli", "daemon-internal-run"],
    )
    monkeypatch.setattr(
        "mcp_markdown_ragdocs.worker.process._read_process_start_time_ticks",
        lambda _pid: 222,
    )

    assert is_expected_daemon_parent(1234, 111) is False


def test_worker_process_start_terminates_runtime_matching_workers(
    monkeypatch,
    tmp_path: Path,
):
    fake_process = _FakeProcess()
    terminated: list[int] = []

    monkeypatch.setattr(
        "mcp_markdown_ragdocs.worker.process._resolve_daemon_python",
        lambda: Path("/repo/.venv/bin/python"),
    )
    monkeypatch.setattr(
        "mcp_markdown_ragdocs.worker.process.current_process_start_time_ticks",
        lambda: None,
    )
    monkeypatch.setattr(
        "mcp_markdown_ragdocs.worker.process._find_runtime_worker_pids",
        lambda _runtime_paths: [111, 222],
    )
    monkeypatch.setattr(
        "mcp_markdown_ragdocs.worker.process.os.getpid",
        lambda: 222,
    )
    monkeypatch.setattr(
        "mcp_markdown_ragdocs.worker.process._terminate_process",
        lambda pid, timeout=1.0: terminated.append(pid),
    )
    monkeypatch.setattr(
        "mcp_markdown_ragdocs.worker.process.subprocess.Popen",
        lambda *args, **kwargs: fake_process,
    )
    monkeypatch.setattr("mcp_markdown_ragdocs.worker.process.WORKER_STARTUP_TIMEOUT_SECONDS", 0.0)

    worker = HueyWorkerProcess(runtime_paths=_paths(tmp_path))
    worker.start()

    assert terminated == [111]


def test_worker_process_status_file_handles_invalid_json_and_cleanup(tmp_path: Path):
    paths = _paths(tmp_path)
    status_path = paths.root / "worker.json"
    status_path.write_text("{invalid", encoding="utf-8")

    assert _read_worker_status(paths) == {}

    status_path.write_text('{"status": "ready"}', encoding="utf-8")
    assert _read_worker_status(paths) == {"status": "ready"}

    _remove_worker_status(paths)
    _remove_worker_status(paths)
    assert not status_path.exists()


def test_worker_process_health_rejects_stale_or_mismatched_status(
    monkeypatch,
    tmp_path: Path,
):
    fake_process = _FakeProcess()
    worker = HueyWorkerProcess(runtime_paths=_paths(tmp_path))
    worker._process = cast(Any, fake_process)

    for status in (
        {"status": "starting", "pid": fake_process.pid, "heartbeat": 100.0},
        {"status": "ready", "pid": fake_process.pid + 1, "heartbeat": 100.0},
        {"status": "ready", "pid": fake_process.pid, "heartbeat": "now"},
    ):
        monkeypatch.setattr(
            "mcp_markdown_ragdocs.worker.process._read_worker_status",
            lambda _paths, status=status: status,
        )
        assert worker.is_healthy() is False

    monkeypatch.setattr(
        "mcp_markdown_ragdocs.worker.process._read_worker_status",
        lambda _paths: {
            "status": "ready",
            "pid": fake_process.pid,
            "heartbeat": 100.0,
        },
    )
    monkeypatch.setattr(
        "mcp_markdown_ragdocs.worker.process.time.time",
        lambda: 401.0,
    )
    assert worker.is_healthy() is False


def _patch_worker_start_for_real_spawn(monkeypatch) -> None:
    monkeypatch.setattr(
        "mcp_markdown_ragdocs.worker.process._resolve_daemon_python",
        lambda: Path(sys.executable),
    )
    monkeypatch.setattr(
        "mcp_markdown_ragdocs.worker.process.current_process_start_time_ticks",
        lambda: None,
    )
    monkeypatch.setattr(
        "mcp_markdown_ragdocs.worker.process._terminate_runtime_worker_processes",
        lambda _runtime_paths: None,
    )
    monkeypatch.setattr("mcp_markdown_ragdocs.worker.process.WORKER_STARTUP_TIMEOUT_SECONDS", 0.0)
    monkeypatch.setattr("mcp_markdown_ragdocs.worker.process.time.sleep", lambda _: None)


def test_worker_process_captures_subprocess_stderr(monkeypatch, tmp_path: Path):
    _patch_worker_start_for_real_spawn(monkeypatch)
    real_popen = subprocess.Popen

    def _fake_popen(_command: list[str], **kwargs: Any):
        script = "import sys; sys.stderr.write('worker-crashed-here\\n')"
        return real_popen([sys.executable, "-c", script], **kwargs)

    monkeypatch.setattr("mcp_markdown_ragdocs.worker.process.subprocess.Popen", _fake_popen)

    worker = HueyWorkerProcess(runtime_paths=_paths(tmp_path))
    worker.start()
    assert worker._process is not None
    worker._process.wait(timeout=5.0)
    worker.stop(timeout=2.0)

    output = (tmp_path / "worker.subprocess.log").read_text(encoding="utf-8")
    assert "worker-crashed-here" in output


def test_worker_process_output_survives_restart(monkeypatch, tmp_path: Path):
    _patch_worker_start_for_real_spawn(monkeypatch)
    real_popen = subprocess.Popen
    run_count = {"n": 0}

    def _fake_popen(_command: list[str], **kwargs: Any):
        run_count["n"] += 1
        script = f"import sys; sys.stderr.write('run-{run_count['n']}-output\\n')"
        return real_popen([sys.executable, "-c", script], **kwargs)

    monkeypatch.setattr("mcp_markdown_ragdocs.worker.process.subprocess.Popen", _fake_popen)

    worker = HueyWorkerProcess(runtime_paths=_paths(tmp_path))
    worker.start()
    assert worker._process is not None
    worker._process.wait(timeout=5.0)

    worker.restart(timeout=2.0)
    assert worker._process is not None
    worker._process.wait(timeout=5.0)
    worker.stop(timeout=2.0)

    output = (tmp_path / "worker.subprocess.log").read_text(encoding="utf-8")
    assert "run-1-output" in output
    assert "run-2-output" in output


def test_worker_process_argument_helpers_match_exact_sequences(tmp_path: Path):
    paths = _paths(tmp_path)
    argv = [
        "python",
        "-m",
        "mcp_markdown_ragdocs.cli",
        "worker-run",
        "--queue-db",
        str(paths.queue_db_path),
        "--index-root",
        str(paths.root),
    ]

    from mcp_markdown_ragdocs.worker.process import (
        _argv_contains_sequence,
        _read_option_value,
    )

    assert _argv_contains_sequence(argv, ("-m", "mcp_markdown_ragdocs.cli", "worker-run"))
    assert not _argv_contains_sequence(argv, ("worker-run", "--index-root"))
    assert _read_option_value(argv, "--queue-db") == str(paths.queue_db_path.resolve())
    assert _read_option_value(argv, "--missing") is None
    assert _read_option_value(["--queue-db"], "--queue-db") is None
