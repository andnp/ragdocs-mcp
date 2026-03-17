from __future__ import annotations

from pathlib import Path

from src.daemon.management import DaemonInspection, start_daemon
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
            DaemonInspection(metadata=None, running=False, stale=False, ready=False),
            DaemonInspection(
                metadata=DaemonMetadata(pid=101, started_at=1.0, status="starting"),
                running=True,
                stale=False,
                ready=False,
            ),
            DaemonInspection(metadata=metadata, running=True, stale=False, ready=True),
        ]
    )

    monkeypatch.setattr("src.daemon.management.inspect_daemon", lambda paths=None: next(inspections))
    monkeypatch.setattr(
        "src.daemon.management._spawn_daemon_process",
        lambda project_override: _FakeProcess(101, [None, None, None]),
    )
    monkeypatch.setattr("src.daemon.management.time.sleep", lambda _: None)

    result = start_daemon(timeout_seconds=0.5, paths=_paths(tmp_path))

    assert result == metadata


def test_start_daemon_accepts_race_winner_metadata(monkeypatch, tmp_path: Path) -> None:
    winner_metadata = DaemonMetadata(pid=303, started_at=2.0, status="ready")
    inspections = iter(
        [
            DaemonInspection(metadata=None, running=False, stale=False, ready=False),
            DaemonInspection(
                metadata=winner_metadata,
                running=True,
                stale=False,
                ready=True,
            ),
        ]
    )

    monkeypatch.setattr("src.daemon.management.inspect_daemon", lambda paths=None: next(inspections))
    monkeypatch.setattr(
        "src.daemon.management._spawn_daemon_process",
        lambda project_override: _FakeProcess(202, [1, 1, 1]),
    )
    monkeypatch.setattr("src.daemon.management.time.sleep", lambda _: None)

    result = start_daemon(timeout_seconds=0.5, paths=_paths(tmp_path))

    assert result == winner_metadata