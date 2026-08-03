from __future__ import annotations

import json
from pathlib import Path

from mcp_markdown_ragdocs.daemon.producer import (
    ProducerMetadata,
    producer_diagnostics,
    producer_is_live,
    read_process_start_time_ticks,
    read_producer_metadata,
    write_producer_metadata,
)


def test_producer_metadata_round_trip_and_diagnostics(tmp_path: Path) -> None:
    path = tmp_path / "producer.json"
    metadata = ProducerMetadata(
        pid=123,
        start_time_ticks=456,
        started_at=789.0,
        stop_reason="restart",
    )

    write_producer_metadata(path, metadata)

    assert read_producer_metadata(path) == metadata
    assert producer_diagnostics(metadata) == {
        "watcher_active": False,
        "producer_pid": 123,
        "producer_started_at": 789.0,
        "stop_reason": "restart",
    }


def test_producer_metadata_rejects_invalid_payload(tmp_path: Path) -> None:
    path = tmp_path / "producer.json"
    path.write_text("{invalid", encoding="utf-8")
    assert read_producer_metadata(path) is None

    path.write_text(json.dumps({"pid": "not-an-int"}), encoding="utf-8")
    assert read_producer_metadata(path) is None
    assert producer_diagnostics(None)["watcher_active"] is False


def test_producer_liveness_requires_pid_and_start_time_match(monkeypatch) -> None:
    metadata = ProducerMetadata(pid=123, start_time_ticks=456, started_at=1.0)
    monkeypatch.setattr("mcp_markdown_ragdocs.daemon.producer.os.kill", lambda *_: None)
    monkeypatch.setattr(
        "mcp_markdown_ragdocs.daemon.producer.read_process_start_time_ticks",
        lambda _pid: 456,
    )
    assert producer_is_live(metadata) is True

    monkeypatch.setattr(
        "mcp_markdown_ragdocs.daemon.producer.read_process_start_time_ticks",
        lambda _pid: 999,
    )
    assert producer_is_live(metadata) is False


def test_producer_liveness_handles_dead_process_and_procfs_parse_errors(
    monkeypatch,
    tmp_path: Path,
) -> None:
    metadata = ProducerMetadata(pid=123, start_time_ticks=456, started_at=1.0)

    def _dead(*_args):
        raise ProcessLookupError

    monkeypatch.setattr("mcp_markdown_ragdocs.daemon.producer.os.kill", _dead)
    assert producer_is_live(metadata) is False
    assert read_process_start_time_ticks(999999) is None

    path = tmp_path / "producer.json"
    path.write_text("[]", encoding="utf-8")
    assert read_producer_metadata(path) is None
