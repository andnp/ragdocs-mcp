from __future__ import annotations

import asyncio
import os
import threading
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from huey import SqliteHuey
from searchkernel.domain import Record, RecordIdentity

from searchkernel.api import TaskSubmissionResult

from mcp_markdown_ragdocs.coordination.queue import QueueRuntime
from mcp_markdown_ragdocs.daemon.request_router import (
    DaemonAdminTaskSubmissionPort,
    DaemonRequestRouterDependencies,
    build_daemon_request_handler,
)
from mcp_markdown_ragdocs.lifecycle import LifecycleState


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
    search_use_case: object | None = None

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
            last_query_execution_stats=None,
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
        min_score: float | None = None,
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
        if min_score is not None:
            self.query_calls[-1]["min_score"] = min_score
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
    task_submission: DaemonAdminTaskSubmissionPort | None = None,
) -> DaemonRequestRouterDependencies:
    if task_submission is None:
        task_submission = MagicMock(spec=DaemonAdminTaskSubmissionPort)
    queue_path = Path("/tmp") / f"mcp-ragdocs-router-{os.getpid()}.db"
    return DaemonRequestRouterDependencies(
        ctx=ctx,
        coordinator=coordinator,
        runtime_root=Path("/runtime"),
        queue_runtime=QueueRuntime(
            huey=SqliteHuey(
                name="router-test",
                filename=str(queue_path),
                immediate=False,
            ),
            db_path=queue_path,
        ),
        socket_path=Path("/runtime/daemon.sock"),
        index_db_path=Path("/runtime/index.db"),
        get_worker_running=lambda: True,
        get_worker_pid=lambda: 123,
        build_admin_overview_payload=lambda ctx, worker_running, worker_pid, lifecycle: {
            "status": "ok",
            "lifecycle": lifecycle,
            "worker_running": worker_running,
            "worker_pid": worker_pid,
        },
        build_index_stats_payload=lambda ctx: {"status": "ok", "indexed_documents": 1},
        build_queue_status_payload=lambda queue_path, worker_running, backpressure_limit: {
            "status": "ok",
            "queue_db_path": str(queue_path),
            "worker_running": worker_running,
            "backpressure_limit": backpressure_limit,
        },
        submit_record_batch=submit_record_batch,
        task_submission=task_submission,
    )


def _record_payload(source_id: str = "note:1") -> dict[str, object]:
    now = datetime(2026, 7, 29, 12, 30, tzinfo=UTC)
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
    """Build the expensive admin overview without blocking the event loop."""
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
    }


@pytest.mark.asyncio
async def test_admin_overview_route_runs_builder_off_event_loop() -> None:
    """A blocked overview builder must not prevent other async work."""
    ctx = _FakeContext(ready=True)
    coordinator = _FakeCoordinator()
    started = threading.Event()
    release = threading.Event()

    def build_overview(*_args: object) -> dict[str, object]:
        started.set()
        assert release.wait(timeout=2)
        return {"status": "ok"}

    dependencies = replace(
        _build_dependencies(ctx, coordinator),
        build_admin_overview_payload=build_overview,
    )
    handler = build_daemon_request_handler(dependencies)
    task = asyncio.create_task(handler("/api/admin/overview", {}))

    try:
        await asyncio.wait_for(asyncio.to_thread(started.wait, 2), timeout=3)
        assert not task.done()
    finally:
        release.set()

    assert await task == {"status": "ok"}


@pytest.mark.asyncio
async def test_admin_index_route_includes_queue_counts() -> None:
    ctx = _FakeContext(ready=True)
    coordinator = _FakeCoordinator()
    dependencies = replace(
        _build_dependencies(ctx, coordinator),
        build_queue_status_payload=lambda **_: {
            "pending_count": 2,
            "running_count": 1,
            "failed_count": 3,
        },
    )
    handler = build_daemon_request_handler(dependencies)

    payload = await handler("/api/admin/index", {})

    assert payload == {
        "status": "ok",
        "indexed_documents": 1,
        "pending_count": 2,
        "running_count": 1,
        "failed_count": 3,
    }


@pytest.mark.asyncio
async def test_admin_index_route_reports_queued_partial_progress() -> None:
    ctx = _FakeContext(ready=True)
    coordinator = _FakeCoordinator()
    dependencies = replace(
        _build_dependencies(ctx, coordinator),
        build_index_stats_payload=lambda _: {
            "status": "ok",
            "remaining_estimate": 0,
            "index_state": {"status": "partial"},
        },
        build_queue_status_payload=lambda **_: {
            "pending_count": 14,
            "running_count": 1,
            "failed_count": 0,
        },
    )
    handler = build_daemon_request_handler(dependencies)

    payload = await handler("/api/admin/index", {})

    assert payload["remaining_estimate"] == 15
    assert payload["index_state"] == {"status": "partial"}


@pytest.mark.asyncio
async def test_rebuild_status_route_returns_extended_telemetry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = {
        "status": "running",
        "phase": "indexing_documents",
        "documents_completed": 2,
        "documents_total": 4,
        "current_document_path": "/docs/two.md",
        "elapsed_seconds": 3.5,
    }
    monkeypatch.setattr(
        "mcp_markdown_ragdocs.daemon.request_router.read_rebuild_status",
        lambda runtime_root: expected
        if runtime_root == Path("/runtime")
        else {},
    )
    handler = build_daemon_request_handler(
        _build_dependencies(_FakeContext(), _FakeCoordinator())
    )

    payload = await handler("/api/admin/rebuild/status", {})

    assert payload == expected


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
        "query_execution_stats": {},
    }
    assert ctx.ensure_fresh_indices_calls == 0
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
async def test_search_query_route_forwards_raw_min_score() -> None:
    ctx = _FakeContext(ready=True)
    coordinator = _FakeCoordinator()
    handler = build_daemon_request_handler(_build_dependencies(ctx, coordinator))

    await handler(
        "/api/search/query",
        {"query": "low confidence", "min_score": 0.025},
    )

    assert ctx.query_calls[0]["min_score"] == 0.025


@pytest.mark.asyncio
async def test_search_query_route_returns_current_use_case_diagnostics() -> None:
    ctx = _FakeContext(ready=True)
    ctx.orchestrator.last_query_execution_stats = {
        "degraded": False,
        "failures": [],
    }

    async def _execute(_request):
        return SimpleNamespace(
            results=[],
            compression_stats=SimpleNamespace(
                to_dict=lambda: {"after_dedup": 0},
            ),
            strategy_stats=SimpleNamespace(
                to_dict=lambda: {"vector_count": 0},
            ),
            query_execution_stats={
                "degraded": True,
                "failures": ["vector unavailable"],
            },
        )

    ctx.search_use_case = SimpleNamespace(execute=_execute)
    handler = build_daemon_request_handler(_build_dependencies(ctx, _FakeCoordinator()))

    payload = await handler("/api/search/query", {"query": "degraded search"})

    assert payload["query_execution_stats"] == {
        "degraded": True,
        "failures": ["vector unavailable"],
    }


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
    assert all(record.created_at.tzinfo == UTC for record in ctx.indexed_records)


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


@pytest.mark.asyncio
async def test_admin_tasks_purge_route_requires_confirmation() -> None:
    ctx = _FakeContext(ready=True)
    coordinator = _FakeCoordinator()
    handler = build_daemon_request_handler(_build_dependencies(ctx, coordinator))

    payload = await handler("/api/admin/tasks/purge", {"state": "all"})

    assert payload == {
        "status": "error",
        "error": "queue_purge_confirmation_required",
        "details": "Refusing to purge queue state without confirm=true.",
    }


@pytest.mark.asyncio
async def test_admin_tasks_purge_route_uses_huey_storage_helpers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ctx = _FakeContext(ready=True)
    coordinator = _FakeCoordinator()
    dependencies = _build_dependencies(ctx, coordinator)
    handler = build_daemon_request_handler(dependencies)

    observed: dict[str, object] = {}
    def _purge(huey, *, state, worker_running, backpressure_limit):
        observed["huey"] = huey
        return SimpleNamespace(
            to_dict=lambda: {
                "purged_state": state,
                "purged_counts": {
                    "pending": 0,
                    "scheduled": 1,
                    "failed": 0,
                },
                "pending_count": 2,
                "scheduled_count": 0,
                "running_count": 0,
                "failed_count": 0,
                "worker_running": worker_running,
                "backpressure_limit": backpressure_limit,
                "backpressure_utilization": 0.4,
                "task_counts": {},
                "recent_failures": [],
                "pending_tasks": [],
                "scheduled_tasks": [],
            }
        )

    monkeypatch.setattr(
        "mcp_markdown_ragdocs.daemon.request_router.purge_queue_state",
        _purge,
    )

    payload = await handler(
        "/api/admin/tasks/purge",
        {"state": "scheduled", "confirm": True},
    )

    assert observed["huey"] is not None
    assert payload == {
        "status": "ok",
        "queue_db_path": str(dependencies.queue_runtime.db_path),
        "purged_state": "scheduled",
        "purged_counts": {
            "pending": 0,
            "scheduled": 1,
            "failed": 0,
        },
        "pending_count": 2,
        "scheduled_count": 0,
        "running_count": 0,
        "failed_count": 0,
        "worker_running": True,
        "backpressure_limit": 5,
        "backpressure_utilization": 0.4,
        "task_counts": {},
        "recent_failures": [],
        "pending_tasks": [],
        "scheduled_tasks": [],
    }


@pytest.mark.asyncio
async def test_rebuild_submit_ignores_abandoned_running_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    An abandoned rebuild leaves rebuild-status.json stuck at "running"
    with nothing to reset it. Submission must not honour that file over
    the live writer-lease evidence that a resubmission actually found.
    """
    ctx = _FakeContext(ready=True)
    submission = MagicMock(spec=DaemonAdminTaskSubmissionPort)
    submission.submit_rebuild_request.return_value = TaskSubmissionResult(status="enqueued")
    dependencies = _build_dependencies(
        ctx,
        _FakeCoordinator(),
        task_submission=submission,
    )
    handler = build_daemon_request_handler(dependencies)

    monkeypatch.setattr(
        "mcp_markdown_ragdocs.daemon.request_router.read_rebuild_status",
        lambda runtime_root: {"status": "running", "phase": "indexing_git"},
    )
    monkeypatch.setattr(
        "mcp_markdown_ragdocs.daemon.request_router.submit_rebuild_status",
        lambda runtime_root, *, request_id, scope: {"status": "queued"},
    )

    payload = await handler("/api/admin/rebuild/submit", {})

    assert payload["accepted"] is True
    assert payload["already_running"] is False


@pytest.mark.asyncio
async def test_rebuild_submit_still_blocks_concurrent_live_rebuild(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    A genuinely in-flight rebuild holds a heartbeated writer lease, so
    submit_rebuild_request reports backpressure. A concurrent submission
    must still be rejected rather than accepted.
    """
    ctx = _FakeContext(ready=True)
    submission = MagicMock(spec=DaemonAdminTaskSubmissionPort)
    submission.submit_rebuild_request.return_value = TaskSubmissionResult(
        status="backpressured"
    )
    dependencies = _build_dependencies(
        ctx,
        _FakeCoordinator(),
        task_submission=submission,
    )
    handler = build_daemon_request_handler(dependencies)

    monkeypatch.setattr(
        "mcp_markdown_ragdocs.daemon.request_router.read_rebuild_status",
        lambda runtime_root: {"status": "idle"},
    )

    payload = await handler("/api/admin/rebuild/submit", {})

    assert payload["accepted"] is False
    assert payload["retry_later"] is True


class _FakeRecordStorage:
    def __init__(self, identities: list[RecordIdentity]) -> None:
        self._identities = identities
        self.deleted_keys: list[str] | None = None

    def iter_identities(self, *, source_kind=None, status=None):
        return iter(
            identity for identity in self._identities if identity.source_kind == source_kind
        )

    def delete(self, storage_keys) -> None:
        self.deleted_keys = list(storage_keys)


def _attach_record_storage(
    ctx: _FakeContext, storage: _FakeRecordStorage, persist_calls: list[int]
) -> None:
    ctx.index_manager.storage = storage
    ctx.index_manager.persist = lambda: persist_calls.append(1)


@pytest.mark.asyncio
async def test_admin_records_purge_route_requires_workspace_and_source_kind() -> None:
    ctx = _FakeContext(ready=True)
    handler = build_daemon_request_handler(_build_dependencies(ctx, _FakeCoordinator()))

    payload = await handler("/api/admin/records/purge", {"source_kind": "git_commit"})

    assert payload == {"status": "error", "error": "workspace_id_required"}


@pytest.mark.asyncio
async def test_admin_records_purge_route_previews_without_deleting() -> None:
    ctx = _FakeContext(ready=True)
    storage = _FakeRecordStorage(
        [
            RecordIdentity("target-project", "git_commit", "git:abc:summary:0"),
            RecordIdentity("other-project", "git_commit", "git:def:summary:0"),
            RecordIdentity("target-project", "note", "note-1"),
        ]
    )
    _attach_record_storage(ctx, storage, [])
    handler = build_daemon_request_handler(_build_dependencies(ctx, _FakeCoordinator()))

    payload = await handler(
        "/api/admin/records/purge",
        {"workspace_id": "target-project", "source_kind": "git_commit"},
    )

    assert payload == {
        "status": "ok",
        "would_delete": 1,
        "workspace_id": "target-project",
        "source_kind": "git_commit",
    }
    assert storage.deleted_keys is None


@pytest.mark.asyncio
async def test_admin_records_purge_route_deletes_and_persists_when_confirmed() -> None:
    ctx = _FakeContext(ready=True)
    target = RecordIdentity("target-project", "git_commit", "git:abc:summary:0")
    storage = _FakeRecordStorage(
        [target, RecordIdentity("other-project", "git_commit", "git:def:summary:0")]
    )
    persist_calls: list[int] = []
    _attach_record_storage(ctx, storage, persist_calls)
    handler = build_daemon_request_handler(_build_dependencies(ctx, _FakeCoordinator()))

    payload = await handler(
        "/api/admin/records/purge",
        {"workspace_id": "target-project", "source_kind": "git_commit", "confirm": True},
    )

    assert payload == {
        "status": "ok",
        "deleted": 1,
        "workspace_id": "target-project",
        "source_kind": "git_commit",
    }
    assert storage.deleted_keys == [target.storage_key]


class _FakeGitDiffStorage:
    def __init__(self, records: list[Record]) -> None:
        self._by_key = {
            RecordIdentity(
                record.workspace_id, record.source_kind, record.source_id
            ).storage_key: record
            for record in records
        }
        self.deleted_keys: list[str] | None = None

    def iter_identities(self, *, source_kind=None, status=None):
        return iter(
            RecordIdentity(record.workspace_id, record.source_kind, record.source_id)
            for record in self._by_key.values()
            if source_kind is None or record.source_kind == source_kind
        )

    def hydrate_records(self, identities):
        return {
            identity.storage_key: self._by_key.get(identity.storage_key)
            for identity in identities
        }

    def delete(self, storage_keys) -> None:
        self.deleted_keys = list(storage_keys)


def _git_diff_record(
    *, workspace_id: str, source_id: str, timestamp: int
) -> Record:
    now = datetime(2026, 1, 1, tzinfo=UTC)
    return Record(
        source_kind="git_commit",
        source_id=source_id,
        workspace_id=workspace_id,
        title="Commit",
        body="Body",
        created_at=now,
        updated_at=now,
        metadata={"timestamp": timestamp},
    )


def _attach_git_diff_storage(
    ctx: _FakeContext,
    storage: _FakeGitDiffStorage,
    persist_calls: list[int],
    *,
    max_age_days: int = 30,
) -> None:
    ctx.index_manager.storage = storage
    ctx.index_manager.persist = lambda: persist_calls.append(1)
    ctx.config.git_indexing = SimpleNamespace(git_diff_embedding_days=max_age_days)


@pytest.mark.asyncio
async def test_prune_old_git_diffs_requires_workspace_id() -> None:
    ctx = _FakeContext(ready=True)
    _attach_git_diff_storage(ctx, _FakeGitDiffStorage([]), [])
    handler = build_daemon_request_handler(_build_dependencies(ctx, _FakeCoordinator()))

    payload = await handler("/api/admin/records/prune-old-git-diffs", {})

    assert payload == {"status": "error", "error": "workspace_id_required"}


@pytest.mark.asyncio
async def test_prune_old_git_diffs_previews_stale_diffs_without_deleting() -> None:
    now = datetime.now(UTC)
    reference_recent = int((now - timedelta(days=1)).timestamp())
    reference_old = int((now - timedelta(days=400)).timestamp())
    old_diff = _git_diff_record(
        workspace_id="ws", source_id="git:aaa:diff:0", timestamp=reference_old
    )
    old_summary = _git_diff_record(
        workspace_id="ws", source_id="git:aaa:summary:0", timestamp=reference_old
    )
    recent_diff = _git_diff_record(
        workspace_id="ws", source_id="git:bbb:diff:0", timestamp=reference_recent
    )
    other_workspace_diff = _git_diff_record(
        workspace_id="other", source_id="git:ccc:diff:0", timestamp=reference_old
    )
    near_miss = _git_diff_record(
        workspace_id="ws", source_id="git:ddd:diffs:0", timestamp=reference_old
    )
    non_git = Record(
        source_kind="note",
        source_id="note:1",
        workspace_id="ws",
        title="Note",
        body="Body",
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        updated_at=datetime(2026, 1, 1, tzinfo=UTC),
    )
    storage = _FakeGitDiffStorage(
        [old_diff, old_summary, recent_diff, other_workspace_diff, near_miss, non_git]
    )
    ctx = _FakeContext(ready=True)
    _attach_git_diff_storage(ctx, storage, [], max_age_days=30)
    handler = build_daemon_request_handler(_build_dependencies(ctx, _FakeCoordinator()))

    payload = await handler(
        "/api/admin/records/prune-old-git-diffs", {"workspace_id": "ws"}
    )

    assert payload == {
        "status": "ok",
        "would_delete": 1,
        "workspace_id": "ws",
        "max_age_days": 30,
    }
    assert storage.deleted_keys is None


@pytest.mark.asyncio
async def test_prune_old_git_diffs_deletes_and_persists_when_confirmed() -> None:
    reference_old = int(datetime(2025, 1, 1, tzinfo=UTC).timestamp())
    old_diff = _git_diff_record(
        workspace_id="ws", source_id="git:aaa:diff:0", timestamp=reference_old
    )
    storage = _FakeGitDiffStorage([old_diff])
    ctx = _FakeContext(ready=True)
    persist_calls: list[int] = []
    _attach_git_diff_storage(ctx, storage, persist_calls, max_age_days=30)
    handler = build_daemon_request_handler(_build_dependencies(ctx, _FakeCoordinator()))

    payload = await handler(
        "/api/admin/records/prune-old-git-diffs",
        {"workspace_id": "ws", "confirm": True},
    )

    expected_key = RecordIdentity("ws", "git_commit", "git:aaa:diff:0").storage_key
    assert payload == {
        "status": "ok",
        "deleted": 1,
        "workspace_id": "ws",
        "max_age_days": 30,
    }
    assert storage.deleted_keys == [expected_key]
    assert persist_calls == [1]
