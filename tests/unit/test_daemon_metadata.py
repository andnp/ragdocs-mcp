from pathlib import Path

from src.daemon.metadata import (
    DaemonMetadata,
    read_daemon_metadata,
    remove_daemon_metadata,
    write_daemon_metadata,
)


def test_write_and_read_daemon_metadata_round_trip(tmp_path: Path) -> None:
    metadata_path = tmp_path / "daemon.json"
    metadata = DaemonMetadata(
        pid=1234,
        started_at=1_763_700_000.0,
        status="ready",
        socket_path="/tmp/ragdocs.sock",
        binary_path="/usr/bin/python",
        version="2.0.0",
        index_db_path="/tmp/index.db",
        queue_db_path="/tmp/queue.db",
    )

    write_daemon_metadata(metadata_path, metadata)
    loaded = read_daemon_metadata(metadata_path)

    assert loaded == metadata
    assert loaded is not None
    assert loaded.transport_endpoint == "ipc:///tmp/ragdocs.sock"


def test_read_daemon_metadata_tolerates_unknown_fields(tmp_path: Path) -> None:
    metadata_path = tmp_path / "daemon.json"
    metadata_path.write_text(
        '{"pid": 1, "started_at": 2.0, "status": "ready", "unknown": true}',
        encoding="utf-8",
    )

    loaded = read_daemon_metadata(metadata_path)

    assert loaded is not None
    assert loaded.pid == 1
    assert loaded.transport == "zmq"
    assert loaded.daemon_scope == "global"


def test_remove_daemon_metadata_is_idempotent(tmp_path: Path) -> None:
    metadata_path = tmp_path / "daemon.json"
    metadata_path.write_text("{}", encoding="utf-8")

    remove_daemon_metadata(metadata_path)
    remove_daemon_metadata(metadata_path)

    assert not metadata_path.exists()