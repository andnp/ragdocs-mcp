from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.daemon.request_router import (
    DaemonRequestRouterDependencies,
    build_daemon_request_handler,
)
from src.lifecycle import LifecycleState


@dataclass
class _FakeIndexState:
    status: str = "indexing"
    indexed_count: int = 0
    total_count: int = 0
    last_error: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "indexed_count": self.indexed_count,
            "total_count": self.total_count,
            "last_error": self.last_error,
        }


@dataclass
class _FakeContext:
    ready: bool = False
    ensure_fresh_indices_calls: int = 0
    schedule_freshness_refresh_calls: int = 0
    documents_roots: list[Path] = field(default_factory=lambda: [Path("/docs")])
    index_state: _FakeIndexState = field(default_factory=_FakeIndexState)

    def __post_init__(self) -> None:
        self.config = SimpleNamespace(
            indexing=SimpleNamespace(task_backpressure_limit=5),
        )
        self.commit_indexer = None
        self.orchestrator = SimpleNamespace(
            query=self._query,
            drain_reindex=self._drain_reindex,
        )
        self.query_calls: list[dict[str, object]] = []
        self.drain_reindex_calls = 0

    def is_ready(self) -> bool:
        return self.ready

    def get_index_state(self) -> _FakeIndexState:
        return self.index_state

    async def ensure_fresh_indices(self) -> None:
        self.ensure_fresh_indices_calls += 1

    def schedule_freshness_refresh(self) -> bool:
        self.schedule_freshness_refresh_calls += 1
        return True

    async def _query(self, query: str, *, top_k: int, top_n: int, project_filter, project_context):
        self.query_calls.append(
            {
                "query": query,
                "top_k": top_k,
                "top_n": top_n,
                "project_filter": project_filter,
                "project_context": project_context,
            }
        )
        return [], SimpleNamespace(to_dict=lambda: {"after_dedup": 0}), SimpleNamespace(to_dict=lambda: {"vector_count": 0})

    async def _drain_reindex(self) -> None:
        self.drain_reindex_calls += 1


class _FakeCoordinator:
    def __init__(self) -> None:
        self.state = LifecycleState.READY
        self.shutdown_requested = False

    def request_shutdown(self) -> None:
        self.shutdown_requested = True


def _build_dependencies(ctx: _FakeContext, coordinator: _FakeCoordinator) -> DaemonRequestRouterDependencies:
    return DaemonRequestRouterDependencies(
        ctx=ctx,
        coordinator=coordinator,
        runtime_root=Path("/runtime"),
        queue_db_path=Path("/runtime/queue.db"),
        socket_path=Path("/runtime/daemon.sock"),
        index_db_path=Path("/runtime/index.db"),
        get_worker_running=lambda: True,
        get_worker_pid=lambda: 123,
        build_admin_overview_payload=lambda ctx, runtime_root, worker_running, worker_pid, lifecycle: {
            "status": "ok",
            "lifecycle": lifecycle,
            "worker_running": worker_running,
            "worker_pid": worker_pid,
            "runtime_root": str(runtime_root),
        },
        build_index_stats_payload=lambda ctx: {"status": "ok", "indexed_documents": 1},
        build_queue_status_payload=lambda queue_path, worker_running, backpressure_limit: {
            "status": "ok",
            "queue_db_path": str(queue_path),
            "worker_running": worker_running,
            "backpressure_limit": backpressure_limit,
        },
    )


@pytest.mark.asyncio
async def test_admin_overview_route_refreshes_indices_before_building_payload() -> None:
    ctx = _FakeContext(ready=True)
    coordinator = _FakeCoordinator()
    handler = build_daemon_request_handler(_build_dependencies(ctx, coordinator))

    payload = await handler("/api/admin/overview", {})

    assert ctx.ensure_fresh_indices_calls == 1
    assert payload == {
        "status": "ok",
        "lifecycle": "ready",
        "worker_running": True,
        "worker_pid": 123,
        "runtime_root": "/runtime",
    }


@pytest.mark.asyncio
async def test_search_query_route_returns_initializing_payload_while_cold() -> None:
    ctx = _FakeContext(ready=False)
    coordinator = _FakeCoordinator()
    coordinator.state = LifecycleState.INITIALIZING
    handler = build_daemon_request_handler(_build_dependencies(ctx, coordinator))

    payload = await handler("/api/search/query", {"query": "startup"})

    assert payload["status"] == "initializing"
    assert payload["query"] == "startup"
    assert payload["lifecycle"] == "initializing"
    assert ctx.schedule_freshness_refresh_calls == 0


@pytest.mark.asyncio
async def test_search_query_route_executes_query_when_ready() -> None:
    ctx = _FakeContext(ready=True)
    coordinator = _FakeCoordinator()
    handler = build_daemon_request_handler(_build_dependencies(ctx, coordinator))

    payload = await handler(
        "/api/search/query",
        {
            "query": "daemon transport",
            "top_n": 3,
            "project_filter": ["proj-a"],
            "project_context": "proj-a",
        },
    )

    assert payload == {
        "query": "daemon transport",
        "results": [],
        "compression_stats": {"after_dedup": 0},
        "strategy_stats": {"vector_count": 0},
    }
    assert ctx.schedule_freshness_refresh_calls == 1
    assert ctx.drain_reindex_calls == 1
    assert ctx.query_calls == [
        {
            "query": "daemon transport",
            "top_k": 30,
            "top_n": 3,
            "project_filter": ["proj-a"],
            "project_context": "proj-a",
        }
    ]


@pytest.mark.asyncio
async def test_internal_shutdown_route_requests_shutdown() -> None:
    ctx = _FakeContext(ready=True)
    coordinator = _FakeCoordinator()
    handler = build_daemon_request_handler(_build_dependencies(ctx, coordinator))

    payload = await handler("/internal/shutdown", {})

    assert coordinator.shutdown_requested is True
    assert payload == {"status": "ok", "lifecycle": "ready"}