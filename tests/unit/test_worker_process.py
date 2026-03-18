from __future__ import annotations

from pathlib import Path
import subprocess

from src.daemon.paths import RuntimePaths
from src.worker.process import HueyWorkerProcess


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
    observed: dict[str, object] = {}
    fake_process = _FakeProcess()

    monkeypatch.setattr(
        "src.worker.process._resolve_daemon_python",
        lambda: Path("/repo/.venv/bin/python"),
    )

    def _fake_popen(command, **kwargs):
        observed["command"] = command
        observed["kwargs"] = kwargs
        return fake_process

    monkeypatch.setattr("src.worker.process.subprocess.Popen", _fake_popen)

    worker = HueyWorkerProcess(runtime_paths=_paths(tmp_path), project_override="docs")
    worker.start()

    command = observed["command"]
    assert command[:4] == [
        "/repo/.venv/bin/python",
        "-m",
        "src.cli",
        "worker-run",
    ]
    assert "--queue-db" in command
    assert "--index-root" in command
    assert "--parent-pid" in command
    assert "--project" not in command
    assert observed["kwargs"]["start_new_session"] is True
    assert worker.is_running is True


def test_worker_process_stop_sends_sigterm(monkeypatch, tmp_path: Path):
    fake_process = _FakeProcess()

    monkeypatch.setattr(
        "src.worker.process._resolve_daemon_python",
        lambda: Path("/repo/.venv/bin/python"),
    )
    monkeypatch.setattr(
        "src.worker.process.subprocess.Popen",
        lambda *args, **kwargs: fake_process,
    )

    worker = HueyWorkerProcess(runtime_paths=_paths(tmp_path))
    worker.start()
    worker.stop(timeout=2.0)

    assert fake_process.signals == [subprocess.signal.SIGTERM]
    assert worker.is_running is False


def test_worker_process_restart_replaces_process(monkeypatch, tmp_path: Path):
    first = _FakeProcess()
    second = _FakeProcess()
    created = iter([first, second])

    monkeypatch.setattr(
        "src.worker.process._resolve_daemon_python",
        lambda: Path("/repo/.venv/bin/python"),
    )
    monkeypatch.setattr(
        "src.worker.process.subprocess.Popen",
        lambda *args, **kwargs: next(created),
    )
    monkeypatch.setattr("src.worker.process.time.sleep", lambda _: None)

    worker = HueyWorkerProcess(runtime_paths=_paths(tmp_path))
    worker.start()
    assert worker.pid == first.pid

    worker.restart(timeout=2.0)

    assert first.signals == [subprocess.signal.SIGTERM]
    assert worker.pid == second.pid
