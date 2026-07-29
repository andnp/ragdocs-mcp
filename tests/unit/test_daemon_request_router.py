from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from dataclasses import dataclass, field
from pathlib import Path
import threading
from types import SimpleNamespace

import pytest

from searchkernel.daemon.request_router import (
    DaemonRequestRouterDependencies,
    build_daemon_request_handler,
)
from searchkernel.domain import Record
from searchkernel.lifecycle import LifecycleState


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
        self.git_indexing_enabled = False
        self.indexed_records: list[Record] = []

        def _index_record(record: Record) -> None:
            self.indexed_records.append(record)

        self.index_manager = SimpleNamespace(index_record=_index_record)
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

    async def _query(
        self,
        query: str,
        *,
        top_k: int,
        top_n: int,
        project_filter,
        source_filter,
        project_context,
    ):
        self.query_calls.append(
            {
                "query": query,
                "top_k": top_k,
                "top_n": top_n,
                "project_filter": project_filter,
                "source_filter": source_filter,
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


def _build_dependencies(
    ctx: _FakeContext,
    coordinator: _FakeCoordinator,
    *,
    submit_record_batch=None,
) -> DaemonRequestRouterDependencies:
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
        submit_record_batch=submit_record_batch,
    )


def _record_payload(source_id: str = "note:1") -> dict[str, object]:
    now = datetime(2026, 7, 29, 12, 30, tzinfo=timezone.utc)
    return Record(
        source_kind="note",
        source_id=source_id,
        title="A note",
        body="Some note content.",
        created_at=now,
        updated_at=now,
    ).to_dict()


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
            "source_filter": ["git_commit"],
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
            "source_filter": ["git_commit"],
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


@pytest.mark.asyncio
async def test_record_index_route_indexes_deserialized_records_through_live_manager() -> None:
    ctx = _FakeContext(ready=True)
    handler = build_daemon_request_handler(_build_dependencies(ctx, _FakeCoordinator()))

    payload = await handler(
        "/api/index/records",
        {"records": [_record_payload("note:1"), _record_payload("note:2")]},
    )

    assert payload == {"status": "ok", "indexed_count": 2}
    assert [record.source_id for record in ctx.indexed_records] == ["note:1", "note:2"]
    assert all(record.created_at.tzinfo == timezone.utc for record in ctx.indexed_records)


@pytest.mark.asyncio
async def test_record_index_route_can_wait_for_worker_result_without_daemon_indexing() -> None:
    ctx = _FakeContext(ready=True)
    submitted: list[list[dict[str, object]]] = []

    class _FakeResult:
        def get(self, *, blocking: bool, timeout: float) -> dict[str, object]:
            assert blocking is True
            assert timeout == 300.0
            return {"status": "ok", "indexed_count": 2}

    def _submit(records: list[dict[str, object]]) -> _FakeResult:
        submitted.append(records)
        return _FakeResult()

    handler = build_daemon_request_handler(
        _build_dependencies(
            ctx,
            _FakeCoordinator(),
            submit_record_batch=_submit,
        )
    )

    payload = await handler(
        "/api/index/records",
        {"records": [_record_payload("note:1"), _record_payload("note:2")]},
    )

    assert payload == {"status": "ok", "indexed_count": 2}
    assert [record["source_id"] for record in submitted[0]] == ["note:1", "note:2"]
    assert ctx.indexed_records == []


@pytest.mark.asyncio
async def test_record_index_route_does_not_block_other_requests() -> None:
    ctx = _FakeContext(ready=True)
    indexing_started = threading.Event()
    release_indexing = threading.Event()

    def _slow_index_record(record: Record) -> None:
        indexing_started.set()
        assert release_indexing.wait(timeout=1.0)
        ctx.indexed_records.append(record)

    ctx.index_manager.index_record = _slow_index_record
    handler = build_daemon_request_handler(_build_dependencies(ctx, _FakeCoordinator()))

    indexing_task = asyncio.create_task(
        handler(
            "/api/index/records",
            {"records": [_record_payload("note:slow")]},
        )
    )
    await asyncio.to_thread(indexing_started.wait, 1.0)

    fast_response = await asyncio.wait_for(
        handler("/internal/shutdown", {}),
        timeout=0.2,
    )

    release_indexing.set()
    indexing_response = await indexing_task

    assert fast_response["status"] == "ok"
    assert indexing_response == {"status": "ok", "indexed_count": 1}


@pytest.mark.asyncio
async def test_record_index_route_validates_batch_before_indexing() -> None:
    ctx = _FakeContext(ready=True)
    handler = build_daemon_request_handler(_build_dependencies(ctx, _FakeCoordinator()))
    invalid = _record_payload("note:bad")
    invalid["created_at"] = "not-a-datetime"

    payload = await handler(
        "/api/index/records",
        {"records": [_record_payload("note:1"), invalid]},
    )

    assert payload == {
        "status": "error",
        "error": "invalid_record",
        "record_index": 1,
        "details": "created_at must be an ISO datetime string",
    }
    assert ctx.indexed_records == []


@pytest.mark.asyncio
async def test_record_index_route_reports_indexing_failure_and_completed_count() -> None:
    ctx = _FakeContext(ready=True)
    calls = 0

    def _index_record(record: Record) -> None:
        nonlocal calls
        if calls == 1:
            raise RuntimeError("embedding unavailable")
        calls += 1
        ctx.indexed_records.append(record)

    ctx.index_manager.index_record = _index_record
    handler = build_daemon_request_handler(_build_dependencies(ctx, _FakeCoordinator()))

    payload = await handler(
        "/api/index/records",
        {"records": [_record_payload("note:1"), _record_payload("note:2")]},
    )

    assert payload == {
        "status": "error",
        "error": "record_indexing_failed",
        "record_index": 1,
        "indexed_count": 1,
        "details": "embedding unavailable",
    }
    assert [record.source_id for record in ctx.indexed_records] == ["note:1"]


@pytest.mark.asyncio
async def test_record_index_route_requires_records_list() -> None:
    ctx = _FakeContext(ready=True)
    handler = build_daemon_request_handler(_build_dependencies(ctx, _FakeCoordinator()))

    payload = await handler("/api/index/records", {})

    assert payload == {
        "status": "error",
        "error": "records_must_be_list",
        "details": "The request payload must contain a records list.",
    }
    assert ctx.indexed_records == []
