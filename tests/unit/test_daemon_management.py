from __future__ import annotations

from pathlib import Path

import pytest

from src.daemon.management import DaemonInspection, DaemonManagementError, start_daemon, stop_daemon
from src.daemon.metadata import DaemonMetadata
from src.daemon.paths import RuntimePaths


class _FakeProcess:
    def __init__(self, pid: int, polls: list[int | None]):
        self.pid = pid
        self._polls = polls
        self.returncode = None

    def poll(self) -> int | None:
        if self._polls:
            self.returncode = self._polls.pop(0)
        return self.returncode


def _paths(tmp_path: Path) -> RuntimePaths:
    return RuntimePaths(
        root=tmp_path,
        index_db_path=tmp_path / "index.db",
        queue_db_path=tmp_path / "queue.db",
        metadata_path=tmp_path / "daemon.json",
        lock_path=tmp_path / "daemon.lock",
        socket_path=tmp_path / "daemon.sock",
    )


def test_start_daemon_waits_for_ready_metadata(monkeypatch, tmp_path: Path) -> None:
    metadata = DaemonMetadata(pid=101, started_at=1.0, status="ready")
    inspections = iter(
        [
            DaemonInspection(
                metadata=None,
                running=False,
                stale=False,
                responsive=False,
                ready=False,
            ),
            DaemonInspection(
                metadata=DaemonMetadata(pid=101, started_at=1.0, status="starting"),
                running=True,
                stale=False,
                responsive=False,
                ready=False,
            ),
            DaemonInspection(
                metadata=metadata,
                running=True,
                stale=False,
                responsive=True,
                ready=True,
            ),
        ]
    )
    fallback = DaemonInspection(
        metadata=metadata,
        running=True,
        stale=False,
        responsive=True,
        ready=True,
    )

    monkeypatch.setattr(
        "src.daemon.management.inspect_daemon",
        lambda paths=None: next(inspections, fallback),
    )
    monkeypatch.setattr(
        "src.daemon.management._spawn_daemon_process",
        lambda project_override, runtime_paths: _FakeProcess(101, [None, None, None]),
    )
    monkeypatch.setattr(
        "src.daemon.management.probe_daemon_socket",
        lambda *args, **kwargs: metadata,
    )
    monkeypatch.setattr("src.daemon.management.time.sleep", lambda _: None)

    result = start_daemon(timeout_seconds=0.5, paths=_paths(tmp_path))

    assert result == metadata


def test_start_daemon_accepts_race_winner_metadata(monkeypatch, tmp_path: Path) -> None:
    winner_metadata = DaemonMetadata(pid=303, started_at=2.0, status="ready")
    inspections = iter(
        [
            DaemonInspection(
                metadata=None,
                running=False,
                stale=False,
                responsive=False,
                ready=False,
            ),
            DaemonInspection(
                metadata=winner_metadata,
                running=True,
                stale=False,
                responsive=True,
                ready=True,
            ),
        ]
    )
    fallback = DaemonInspection(
        metadata=winner_metadata,
        running=True,
        stale=False,
        responsive=True,
        ready=True,
    )

    monkeypatch.setattr(
        "src.daemon.management.inspect_daemon",
        lambda paths=None: next(inspections, fallback),
    )
    monkeypatch.setattr(
        "src.daemon.management._spawn_daemon_process",
        lambda project_override, runtime_paths: _FakeProcess(202, [1, 1, 1]),
    )
    monkeypatch.setattr(
        "src.daemon.management.probe_daemon_socket",
        lambda *args, **kwargs: winner_metadata,
    )
    monkeypatch.setattr("src.daemon.management.time.sleep", lambda _: None)

    result = start_daemon(timeout_seconds=0.5, paths=_paths(tmp_path))

    assert result == winner_metadata


def test_inspect_daemon_requires_successful_probe(monkeypatch, tmp_path: Path) -> None:
    metadata = DaemonMetadata(pid=404, started_at=1.0, status="ready")
    metadata_path = _paths(tmp_path).metadata_path
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(
        '{"pid": 404, "started_at": 1.0, "status": "ready"}',
        encoding="utf-8",
    )

    monkeypatch.setattr("src.daemon.management.is_process_running", lambda pid: True)
    monkeypatch.setattr("src.daemon.management.probe_daemon_socket", lambda *args, **kwargs: None)

    inspection = __import__("src.daemon.management", fromlist=["inspect_daemon"]).inspect_daemon(_paths(tmp_path))

    assert inspection.metadata == metadata
    assert inspection.running is True
    assert inspection.ready is False
    assert inspection.responsive is False


def test_start_daemon_surfaces_log_excerpt_on_spawn_failure(
    monkeypatch,
    tmp_path: Path,
) -> None:
    paths = _paths(tmp_path)
    log_path = paths.root / "daemon.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("traceback line\nroot cause line", encoding="utf-8")

    inspections = iter(
        [
            DaemonInspection(
                metadata=None,
                running=False,
                stale=False,
                responsive=False,
                ready=False,
            ),
            DaemonInspection(
                metadata=None,
                running=False,
                stale=False,
                responsive=False,
                ready=False,
            ),
        ]
    )
    fallback = DaemonInspection(
        metadata=None,
        running=False,
        stale=False,
        responsive=False,
        ready=False,
    )

    monkeypatch.setattr(
        "src.daemon.management.inspect_daemon",
        lambda paths=None: next(inspections, fallback),
    )
    monkeypatch.setattr(
        "src.daemon.management._spawn_daemon_process",
        lambda project_override, runtime_paths: _FakeProcess(909, [7, 7, 7]),
    )
    monkeypatch.setattr("src.daemon.management.time.sleep", lambda _: None)

    with pytest.raises(DaemonManagementError) as exc_info:
        start_daemon(timeout_seconds=0.2, paths=paths)

    message = str(exc_info.value)
    assert "exit code 7" in message
    assert "Daemon log:" in message
    assert "root cause line" in message


def test_start_daemon_waits_for_responsive_existing_ready_daemon(
    monkeypatch,
    tmp_path: Path,
) -> None:
    paths = _paths(tmp_path)
    metadata = DaemonMetadata(pid=515, started_at=1.0, status="ready_primary")
    inspections = iter(
        [
            DaemonInspection(
                metadata=metadata,
                running=True,
                stale=False,
                responsive=False,
                ready=False,
            ),
            DaemonInspection(
                metadata=metadata,
                running=True,
                stale=False,
                responsive=True,
                ready=True,
            ),
        ]
    )

    monkeypatch.setattr(
        "src.daemon.management.inspect_daemon",
        lambda paths=None: next(inspections),
    )
    monkeypatch.setattr("src.daemon.management.time.sleep", lambda _: None)

    stop_calls: list[float] = []
    monkeypatch.setattr(
        "src.daemon.management.stop_daemon",
        lambda *, timeout_seconds, paths=None: stop_calls.append(timeout_seconds),
    )

    result = start_daemon(timeout_seconds=0.5, paths=paths)

    assert result == metadata
    assert stop_calls == []


def test_start_daemon_cleans_up_old_nonresponsive_metadata_before_spawn(
    monkeypatch,
    tmp_path: Path,
) -> None:
    paths = _paths(tmp_path)
    stale_metadata = DaemonMetadata(
        pid=616,
        started_at=1.0,
        status="initializing",
    )
    replacement_metadata = DaemonMetadata(
        pid=717,
        started_at=50.0,
        status="starting",
    )
    cleaned: list[RuntimePaths] = []
    observed: dict[str, object] = {}

    monkeypatch.setattr(
        "src.daemon.management.inspect_daemon",
        lambda paths=None: DaemonInspection(
            metadata=stale_metadata,
            running=True,
            stale=False,
            responsive=False,
            ready=False,
        ),
    )
    monkeypatch.setattr("src.daemon.management.time.time", lambda: 40.0)
    monkeypatch.setattr(
        "src.daemon.management._cleanup_stale_runtime_state",
        lambda runtime_paths: cleaned.append(runtime_paths),
    )
    monkeypatch.setattr(
        "src.daemon.management._spawn_daemon_process",
        lambda project_override, runtime_paths: _FakeProcess(717, [None, None]),
    )

    def _fake_wait_for_ready_daemon(*, deadline, paths, spawned_process=None):
        observed["spawned_pid"] = None if spawned_process is None else spawned_process.pid
        return replacement_metadata

    monkeypatch.setattr(
        "src.daemon.management._wait_for_ready_daemon",
        _fake_wait_for_ready_daemon,
    )

    result = start_daemon(timeout_seconds=0.5, paths=paths)

    assert result == replacement_metadata
    assert cleaned == [paths]
    assert observed["spawned_pid"] == 717


def test_stop_daemon_prefers_internal_shutdown_for_responsive_daemon(
    monkeypatch,
    tmp_path: Path,
) -> None:
    paths = _paths(tmp_path)
    metadata = DaemonMetadata(
        pid=818,
        started_at=1.0,
        status="ready_primary",
        socket_path=str(paths.socket_path),
    )

    monkeypatch.setattr(
        "src.daemon.management.inspect_daemon",
        lambda paths=None: DaemonInspection(
            metadata=metadata,
            running=True,
            stale=False,
            responsive=True,
            ready=True,
        ),
    )

    observed: dict[str, object] = {}

    def _fake_request(socket_path, path, payload, timeout_seconds):
        observed["socket_path"] = socket_path
        observed["path"] = path
        observed["payload"] = payload
        observed["timeout_seconds"] = timeout_seconds
        return {"status": "ok"}

    monkeypatch.setattr("src.daemon.management.request_daemon_socket", _fake_request)

    running = iter([True, False])
    monkeypatch.setattr(
        "src.daemon.management.is_process_running",
        lambda pid: next(running),
    )
    monkeypatch.setattr("src.daemon.management.time.sleep", lambda _: None)

    terminations: list[tuple[int, bool]] = []
    monkeypatch.setattr(
        "src.daemon.management._terminate_process",
        lambda pid, *, force: terminations.append((pid, force)),
    )

    cleaned: list[RuntimePaths] = []
    monkeypatch.setattr(
        "src.daemon.management._cleanup_stale_runtime_state",
        lambda runtime_paths: cleaned.append(runtime_paths),
    )

    result = stop_daemon(timeout_seconds=0.5, paths=paths)

    assert result == metadata
    assert observed["path"] == "/internal/shutdown"
    assert terminations == []
    assert cleaned == [paths]