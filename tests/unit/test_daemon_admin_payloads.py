from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass

from mcp_markdown_ragdocs.daemon import RuntimePaths
from mcp_markdown_ragdocs.daemon import admin_payloads
from mcp_markdown_ragdocs.coordination.queue import build_queue_runtime


@dataclass
class _TestContext:
    config: _TestConfig
    documents_roots: list[Path]
    index_path: Path
    watcher: None = None

    def get_index_state(self) -> _TestIndexState:
        return _TestIndexState()


@dataclass
class _TestConfig:
    indexing: _TestIndexingConfig


@dataclass
class _TestIndexingConfig:
    task_backpressure_limit: int = 7


@dataclass
class _TestIndexState:
    def to_dict(self) -> dict[str, object]:
        return {"status": "ready"}


def _runtime_paths(root: Path) -> RuntimePaths:
    return RuntimePaths(
        root=root,
        index_db_path=root / "index.db",
        queue_db_path=root / "queue.db",
        metadata_path=root / "daemon.json",
        lock_path=root / "daemon.lock",
        socket_path=root / "daemon.sock",
    )


def test_admin_overview_reuses_expensive_snapshot_but_checks_producer_live(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Repeated overviews skip discovery work while retaining live producer checks."""
    calls = {"index": 0, "reindex": 0, "producer": 0}
    ctx = _TestContext(
        config=_TestConfig(indexing=_TestIndexingConfig()),
        documents_roots=[tmp_path / "docs"],
        index_path=tmp_path / "index",
        watcher=None,
    )
    runtime_paths = _runtime_paths(tmp_path / "runtime")

    def build_index(_ctx) -> dict[str, object]:
        calls["index"] += 1
        return {
            "storage": {"status": "ok"},
            "indexed_documents": 4,
            "indexed_chunks": 8,
            "git_commits": 3,
            "git_repositories": 1,
        }

    def build_reindex(_root, _index_path) -> dict[str, object]:
        calls["reindex"] += 1
        return {"status": "idle"}

    def producer_payload(_metadata) -> dict[str, object]:
        calls["producer"] += 1
        return {
            "watcher_active": True,
            "producer_pid": 123,
            "producer_started_at": 1.0,
            "stop_reason": None,
        }

    monkeypatch.setattr(admin_payloads, "_build_index_stats_payload", build_index)
    monkeypatch.setattr(admin_payloads, "reindex_status_payload", build_reindex)
    monkeypatch.setattr(
        admin_payloads,
        "_build_queue_status_payload",
        lambda **_kwargs: {
            "pending_count": 0,
            "scheduled_count": 0,
            "running_count": 0,
            "failed_count": 0,
            "worker_running": True,
            "git_refresh_progress": [],
        },
    )
    monkeypatch.setattr(admin_payloads, "producer_diagnostics", producer_payload)
    monkeypatch.setattr(admin_payloads, "read_producer_metadata", lambda _path: None)

    queue_runtime = build_queue_runtime(runtime_paths.queue_db_path)
    first = admin_payloads._build_admin_overview_payload(
        ctx, runtime_paths, queue_runtime, True, 456, "ready"
    )
    second = admin_payloads._build_admin_overview_payload(
        ctx, runtime_paths, queue_runtime, True, 456, "ready"
    )

    assert first["indexed_documents"] == second["indexed_documents"] == 4
    assert calls == {"index": 1, "reindex": 1, "producer": 2}
    snapshot_status = second["status_snapshot"]
    assert isinstance(snapshot_status, dict)
    assert snapshot_status["stale"] is False
    assert snapshot_status["error"] is None


def test_admin_overview_surfaces_explicit_refresh_failure(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """An explicit refresh failure is visible without discarding the last good overview."""
    ctx = _TestContext(
        config=_TestConfig(indexing=_TestIndexingConfig()),
        documents_roots=[],
        index_path=tmp_path / "index",
        watcher=None,
    )
    runtime_paths = _runtime_paths(tmp_path / "runtime")
    index_calls = 0

    def build_index(_ctx) -> dict[str, object]:
        nonlocal index_calls
        index_calls += 1
        if index_calls == 2:
            raise RuntimeError("index load failed")
        return {
            "storage": {},
            "indexed_documents": 1,
            "indexed_chunks": 1,
            "git_commits": 0,
            "git_repositories": 0,
        }

    monkeypatch.setattr(admin_payloads, "_build_index_stats_payload", build_index)
    monkeypatch.setattr(
        admin_payloads,
        "reindex_status_payload",
        lambda _root, _index_path: {"status": "idle"},
    )
    monkeypatch.setattr(
        admin_payloads,
        "_build_queue_status_payload",
        lambda **_kwargs: {
            "pending_count": 0,
            "scheduled_count": 0,
            "running_count": 0,
            "failed_count": 0,
            "worker_running": True,
            "git_refresh_progress": [],
        },
    )
    monkeypatch.setattr(admin_payloads, "producer_diagnostics", lambda _metadata: {})
    monkeypatch.setattr(admin_payloads, "read_producer_metadata", lambda _path: None)

    queue_runtime = build_queue_runtime(runtime_paths.queue_db_path)
    admin_payloads._build_admin_overview_payload(
        ctx, runtime_paths, queue_runtime, True, 456, "ready"
    )
    refresh_status = admin_payloads.refresh_admin_overview_snapshot(ctx, runtime_paths)
    overview = admin_payloads._build_admin_overview_payload(
        ctx, runtime_paths, queue_runtime, True, 456, "ready"
    )

    assert refresh_status["stale"] is True
    assert refresh_status["error"] == "RuntimeError: index load failed"
    assert overview["indexed_documents"] == 1
    overview_status = overview["status_snapshot"]
    assert isinstance(overview_status, dict)
    assert overview_status["stale"] is True
    assert overview_status["error"] == "RuntimeError: index load failed"
