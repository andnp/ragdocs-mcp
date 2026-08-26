from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass

from mcp_markdown_ragdocs.daemon import RuntimePaths
from mcp_markdown_ragdocs.daemon import admin_payloads
from mcp_markdown_ragdocs.coordination.queue import build_queue_runtime
from mcp_markdown_ragdocs.daemon.queue_status import QueueStats
from mcp_markdown_ragdocs.daemon.status_snapshot import StatusSnapshot


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
    now = 0.0

    def clock() -> float:
        return now

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
    monkeypatch.setattr(
        admin_payloads,
        "_status_snapshot",
        StatusSnapshot(stale_after_seconds=5.0, clock=clock),
    )
    monkeypatch.setattr(admin_payloads, "_status_snapshot_root", runtime_paths.root.resolve())

    queue_runtime = build_queue_runtime(runtime_paths.queue_db_path)
    first = admin_payloads._build_admin_overview_payload(
        ctx, runtime_paths, queue_runtime, True, 456, "ready"
    )
    second = admin_payloads._build_admin_overview_payload(
        ctx, runtime_paths, queue_runtime, True, 456, "ready"
    )
    now = 6.0
    third = admin_payloads._build_admin_overview_payload(
        ctx, runtime_paths, queue_runtime, True, 456, "ready"
    )

    assert first["indexed_documents"] == second["indexed_documents"] == 4
    assert third["indexed_documents"] == 4
    assert calls == {"index": 2, "reindex": 2, "producer": 3}
    snapshot_status = third["status_snapshot"]
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


def test_queue_status_payload_served_from_cache_within_ttl_then_refreshed(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """The expensive queue query is reused within the TTL and re-run once it expires.

    worker_running is intentionally excluded from the cached value and must stay
    live on every call, since callers (e.g. worker_health reporting) depend on it
    reflecting the current process rather than a stale cached snapshot.
    """
    calls = {"n": 0}
    now = 0.0

    def clock() -> float:
        return now

    def fake_get_queue_stats(_huey, *, worker_running: bool, backpressure_limit):
        calls["n"] += 1
        return QueueStats(
            pending_count=calls["n"],
            scheduled_count=0,
            worker_running=worker_running,
        )

    db_path = tmp_path / "queue.db"
    queue_runtime = build_queue_runtime(db_path)

    monkeypatch.setattr(admin_payloads, "get_queue_stats", fake_get_queue_stats)
    monkeypatch.setattr(admin_payloads, "list_progress", lambda _parent: [])
    monkeypatch.setattr(
        admin_payloads,
        "_queue_status_snapshot",
        StatusSnapshot(stale_after_seconds=5.0, clock=clock),
    )
    monkeypatch.setattr(admin_payloads, "_queue_status_snapshot_root", db_path.resolve())

    first = admin_payloads._build_queue_status_payload(
        queue_runtime=queue_runtime, worker_running=True
    )
    second = admin_payloads._build_queue_status_payload(
        queue_runtime=queue_runtime, worker_running=False
    )

    assert calls["n"] == 1
    assert first["pending_count"] == second["pending_count"] == 1
    assert first["worker_running"] is True
    assert second["worker_running"] is False

    now = 6.0
    third = admin_payloads._build_queue_status_payload(
        queue_runtime=queue_runtime, worker_running=True
    )

    assert calls["n"] == 2
    assert third["pending_count"] == 2
    assert third["worker_running"] is True
    queue_runtime.huey.storage.close()


def test_queue_status_payload_shape_is_unchanged(tmp_path: Path) -> None:
    """Callers (CLI, admin overview) key off this exact payload shape; keys must not shift."""
    db_path = tmp_path / "queue.db"
    queue_runtime = build_queue_runtime(db_path)

    payload = admin_payloads._build_queue_status_payload(
        queue_runtime=queue_runtime,
        worker_running=True,
        backpressure_limit=10,
    )

    assert set(payload) == {
        "pending_count",
        "scheduled_count",
        "running_count",
        "failed_count",
        "historical_failure_count",
        "worker_running",
        "backpressure_limit",
        "backpressure_utilization",
        "task_counts",
        "recent_failures",
        "pending_tasks",
        "scheduled_tasks",
        "queue_db_path",
        "git_refresh_progress",
    }
    assert payload["worker_running"] is True
    assert payload["pending_count"] == 0
    assert payload["recent_failures"] == []
    queue_runtime.huey.storage.close()
