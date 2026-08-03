from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable, Coroutine
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast
from uuid import uuid4

from mcp_markdown_ragdocs.app.search import SearchQuery
from mcp_markdown_ragdocs.coordination.queue import get_huey
from mcp_markdown_ragdocs.daemon.mcp_requests import (
    build_mcp_tools_payload,
    handle_mcp_tool_call,
)
from mcp_markdown_ragdocs.daemon.queue_status import purge_queue_state
from mcp_markdown_ragdocs.daemon.record_rpc import RecordSerializationError, deserialize_record
from searchkernel.api import Record
from mcp_markdown_ragdocs.indexing.rebuild_service import (
    REBUILD_ACTIVE_STATUSES,
    read_rebuild_status,
    resolve_rebuild_scope,
    submit_rebuild_status,
)
from mcp_markdown_ragdocs.indexing.reindex import (
    REINDEX_ACTIVE_STATUSES,
    read_reindex_status,
    reindex_status_payload,
    submit_reindex_status,
    write_reindex_status,
)
from mcp_markdown_ragdocs.indexing.tasks import (
    submit_rebuild_request,
    submit_reindex_request,
)
from mcp_markdown_ragdocs.models import ChunkResult

logger = logging.getLogger(__name__)


def _as_int(value: object, default: int = 0) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) else default


type BuildAdminOverviewPayload = Callable[..., dict[str, object]]
type BuildIndexStatsPayload = Callable[..., dict[str, object]]
type BuildQueueStatusPayload = Callable[..., dict[str, object]]
type SubmitRecordBatch = Callable[[list[dict[str, object]]], object | None]


class _RecordIndexManager(Protocol):
    def index_record(self, record: Record) -> None: ...


class _RecordIndexContext(Protocol):
    index_manager: _RecordIndexManager


class _SearchPayloadContext(Protocol):
    documents_roots: list[Path]

    def get_index_state(self) -> Any: ...

    def get_total_git_commits_indexed(self) -> int: ...


class _SearchPayloadCoordinator(Protocol):
    state: Any


class _RouterContext(Protocol):
    documents_roots: list[Path]
    index_path: Path
    index_manager: _RecordIndexManager
    orchestrator: Any
    search_use_case: Any
    config: Any
    git_indexing_enabled: bool

    def is_ready(self) -> bool: ...
    def get_index_state(self) -> Any: ...
    async def ensure_fresh_indices(self) -> None: ...
    def schedule_freshness_refresh(self) -> bool: ...
    def get_total_git_commits_indexed(self) -> int: ...


class _RouterCoordinator(Protocol):
    state: Any

    def request_shutdown(self) -> None: ...

    async def wait_ready(self, timeout: float = 60.0) -> None: ...


def _index_records(ctx: _RecordIndexContext, payload: dict[str, object]) -> dict[str, object]:
    records, error = _deserialize_records(payload)
    if error is not None:
        return error

    for index, record in enumerate(records):
        try:
            ctx.index_manager.index_record(record)
        except Exception as exc:  # noqa: BLE001 -- indexing pipeline errors vary; report per-record failure
            return {
                "status": "error",
                "error": "record_indexing_failed",
                "record_index": index,
                "indexed_count": index,
                "details": str(exc),
            }

    return {"status": "ok", "indexed_count": len(records)}


def _deserialize_records(
    payload: dict[str, object],
) -> tuple[list[Record], dict[str, object] | None]:
    raw_records = payload.get("records")
    if not isinstance(raw_records, list):
        return [], {
            "status": "error",
            "error": "records_must_be_list",
            "details": "The request payload must contain a records list.",
        }

    records: list[Record] = []
    for index, raw_record in enumerate(raw_records):
        try:
            records.append(deserialize_record(raw_record))
        except RecordSerializationError as exc:
            return [], {
                "status": "error",
                "error": "invalid_record",
                "record_index": index,
                "details": str(exc),
            }
    return records, None


@dataclass(frozen=True)
class DaemonRequestRouterDependencies:
    ctx: object
    coordinator: object
    runtime_root: Path
    queue_db_path: Path
    socket_path: Path
    index_db_path: Path
    get_worker_running: Callable[[], bool]
    get_worker_pid: Callable[[], int | None]
    build_admin_overview_payload: BuildAdminOverviewPayload
    build_index_stats_payload: BuildIndexStatsPayload
    build_queue_status_payload: BuildQueueStatusPayload
    submit_record_batch: SubmitRecordBatch | None = None


def _record_batch_error(details: str) -> dict[str, object]:
    return {
        "status": "error",
        "error": "record_indexing_failed",
        "details": details,
    }


async def _submit_record_batch(
    dependencies: DaemonRequestRouterDependencies,
    payload: dict[str, object],
) -> dict[str, object]:
    records, error = _deserialize_records(payload)
    if error is not None:
        return error

    if dependencies.submit_record_batch is None:
        return await asyncio.to_thread(
            _index_records,
            cast(_RecordIndexContext, dependencies.ctx),
            payload,
        )

    result = await asyncio.to_thread(
        dependencies.submit_record_batch,
        [record.to_dict() for record in records],
    )
    if result is None:
        return _record_batch_error("Index worker is unavailable.")

    result_get = getattr(result, "get", None)
    if not callable(result_get):
        return _record_batch_error("Index worker returned an invalid result handle.")

    try:
        response = await asyncio.to_thread(
            result_get,
            blocking=True,
            timeout=300.0,
        )
    except TimeoutError:
        return _record_batch_error("Timed out waiting for the index worker.")
    except Exception as exc:
        logger.exception("Record batch task failed")
        return _record_batch_error(str(exc))

    if isinstance(response, dict):
        return response
    return _record_batch_error("Index worker returned an invalid result.")


def _filter_git_history_results(
    results: list,
    files_glob: str | None,
    after_timestamp: int | None,
    before_timestamp: int | None,
) -> list:
    """Post-filter git_commit ChunkResults by files_glob/timestamp bounds.

    SearchOrchestrator.query's source_filter narrows by source_kind only; the
    files_glob/after/before filters git history search has always exposed are
    applied here against the same commit metadata GitContentSource attaches.
    """
    if not (files_glob or after_timestamp is not None or before_timestamp is not None):
        return results

    filtered = []
    for result in results:
        metadata = result.metadata
        timestamp = metadata.get("timestamp")
        if after_timestamp is not None and (timestamp is None or timestamp <= after_timestamp):
            continue
        if before_timestamp is not None and (timestamp is None or timestamp >= before_timestamp):
            continue
        if files_glob:
            files_changed = metadata.get("files_changed") or []
            if not any(Path(f).match(files_glob) for f in files_changed):
                continue
        filtered.append(result)
    return filtered


def _git_history_result_to_dict(result) -> dict[str, object]:
    metadata = result.metadata
    return {
        "hash": result.doc_id.removeprefix("git:"),
        "title": metadata.get("title", ""),
        "author": metadata.get("author", "Unknown"),
        "committer": metadata.get("committer", "Unknown"),
        "timestamp": metadata.get("timestamp"),
        "message": result.content,
        "files_changed": metadata.get("files_changed") or [],
        "score": result.score,
        "project_id": result.project_id,
    }


def _build_initializing_search_payload(
    ctx: object,
    coordinator: object,
    *,
    query: str,
    include_git_metadata: bool = False,
) -> dict[str, object]:
    search_ctx = cast(_SearchPayloadContext, ctx)
    search_coordinator = cast(_SearchPayloadCoordinator, coordinator)
    payload: dict[str, object] = {
        "status": "initializing",
        "message": "Search indices are still initializing. Retry shortly.",
        "query": query,
        "results": [],
        "lifecycle": search_coordinator.state.value,
        "daemon_scope": "global",
        "project_context_mode": "request_only",
        "configured_root_count": len(search_ctx.documents_roots),
        "index_state": search_ctx.get_index_state().to_dict(),
    }
    if include_git_metadata:
        payload["total_commits_indexed"] = search_ctx.get_total_git_commits_indexed()
    else:
        payload["compression_stats"] = {}
        payload["strategy_stats"] = {}
    return payload


def _build_unavailable_search_payload(
    ctx: object,
    coordinator: object,
) -> dict[str, object]:
    search_ctx = cast(_SearchPayloadContext, ctx)
    search_coordinator = cast(_SearchPayloadCoordinator, coordinator)
    index_state = search_ctx.get_index_state()
    return {
        "status": "error",
        "error": "index_initialization_failed",
        "details": index_state.last_error or "Search indices are not queryable.",
        "lifecycle": search_coordinator.state.value,
        "daemon_scope": "global",
        "project_context_mode": "request_only",
        "configured_root_count": len(search_ctx.documents_roots),
        "index_state": index_state.to_dict(),
    }


def _get_cold_start_search_response(
    ctx: object,
    coordinator: object,
    *,
    query: str,
    include_git_metadata: bool = False,
) -> dict[str, object] | None:
    search_ctx = cast(_RouterContext, ctx)
    if search_ctx.is_ready():
        return None

    index_state = search_ctx.get_index_state()
    if index_state.status in {"failed", "partial"}:
        return _build_unavailable_search_payload(search_ctx, coordinator)

    return _build_initializing_search_payload(
        search_ctx,
        coordinator,
        query=query,
        include_git_metadata=include_git_metadata,
    )




async def _handle_mcp_request(
    path: str,
    payload: dict[str, object],
    ctx: _RouterContext,
    coordinator: _RouterCoordinator,
) -> dict[str, object] | None:
    if path == "/api/mcp/tools":
        return build_mcp_tools_payload()
    if path == "/api/mcp/tool":
        return await handle_mcp_tool_call(
            ctx_getter=lambda: ctx,
            coordinator=coordinator,
            payload=payload,
        )
    return None


async def _handle_admin_request(
    dependencies: DaemonRequestRouterDependencies,
    path: str,
    payload: dict[str, object],
    ctx: _RouterContext,
    coordinator: _RouterCoordinator,
) -> dict[str, object] | None:
    if path == "/api/admin/overview":
        await ctx.ensure_fresh_indices()
        return dependencies.build_admin_overview_payload(
            ctx,
            dependencies.runtime_root,
            dependencies.get_worker_running(),
            dependencies.get_worker_pid(),
            coordinator.state.value,
        )
    if path in {"/api/admin/index", "/api/admin/index-stats"}:
        await ctx.ensure_fresh_indices()
        return dependencies.build_index_stats_payload(ctx)
    if path in {"/api/admin/tasks", "/api/admin/queue-status"}:
        return dependencies.build_queue_status_payload(
            queue_path=dependencies.queue_db_path,
            worker_running=dependencies.get_worker_running(),
            backpressure_limit=ctx.config.indexing.task_backpressure_limit,
        )
    if path == "/api/admin/tasks/purge":
        if payload.get("confirm") is not True:
            return {
                "status": "error",
                "error": "queue_purge_confirmation_required",
                "details": "Refusing to purge queue state without confirm=true.",
            }

        state = str(payload.get("state", "all")).lower()
        if state not in {"pending", "scheduled", "failed", "all"}:
            return {
                "status": "error",
                "error": "invalid_queue_purge_state",
                "details": "state must be one of: pending, scheduled, failed, all",
            }

        purge_result = purge_queue_state(
            get_huey(dependencies.queue_db_path),
            state=state,
            worker_running=dependencies.get_worker_running(),
            backpressure_limit=ctx.config.indexing.task_backpressure_limit,
        )
        response = purge_result.to_dict()
        response["status"] = "ok"
        response["queue_db_path"] = str(dependencies.queue_db_path)
        return response
    if path == "/api/admin/rebuild/status":
        return read_rebuild_status(dependencies.runtime_root)
    if path == "/api/admin/reindex/status":
        return reindex_status_payload(
            dependencies.runtime_root,
            Path(getattr(ctx, "index_path")),
        )
    if path == "/api/admin/reindex/submit":
        operation = str(payload.get("operation", "start")).lower()
        if operation not in {"start", "contract", "rollback"}:
            return {
                "status": "error",
                "error": "invalid_reindex_operation",
                "details": "operation must be start, contract, or rollback",
            }
        if ctx.config.store.backend != "pgvector":
            return {
                "status": "error",
                "error": "reindex_backend_unsupported",
                "details": (
                    "durable model migration requires store.backend = "
                    "'pgvector'; the legacy faiss+sqlite chunk index is "
                    "not model-scoped"
                ),
            }

        current_status = read_reindex_status(dependencies.runtime_root)
        if str(current_status.get("status")) in REINDEX_ACTIVE_STATUSES:
            return {
                "status": "ok",
                "accepted": False,
                "already_running": True,
                "reindex": reindex_status_payload(
                    dependencies.runtime_root,
                    Path(getattr(ctx, "index_path")),
                ),
            }

        model = (
            str(payload.get("model"))
            if payload.get("model") is not None
            else None
        )
        old_model = (
            str(payload.get("old_model"))
            if payload.get("old_model") is not None
            else None
        )
        raw_truncate_dim = payload.get("truncate_dim")
        truncate_dim = (
            int(raw_truncate_dim)
            if isinstance(raw_truncate_dim, int)
            and not isinstance(raw_truncate_dim, bool)
            else None
        )
        if operation == "start" and not model:
            return {
                "status": "error",
                "error": "reindex_model_required",
                "details": "model is required for a start operation",
            }

        request_id = uuid4().hex
        queued_status = submit_reindex_status(
            dependencies.runtime_root,
            operation=operation,
            request_id=request_id,
            model=model,
            truncate_dim=truncate_dim,
            old_model=old_model,
        )
        submission = submit_reindex_request(
            operation,
            model=model,
            truncate_dim=truncate_dim,
            old_model=old_model,
            request_id=request_id,
        )
        if not submission.queue_available:
            write_reindex_status(
                dependencies.runtime_root,
                {"status": "idle", "error": "reindex_queue_unavailable"},
            )
            return {
                "status": "error",
                "error": "reindex_queue_unavailable",
                "details": "Index worker is unavailable.",
            }
        if submission.should_retry_later:
            write_reindex_status(
                dependencies.runtime_root,
                {"status": "idle", "error": "reindex_queue_backpressured"},
            )
            return {
                "status": "error",
                "error": "reindex_queue_backpressured",
                "details": "Index worker queue is backpressured.",
            }
        return {
            "status": "ok",
            "accepted": submission.accepted_by_queue,
            "already_running": False,
            "reindex": queued_status,
        }
    if path == "/api/admin/rebuild/submit":
        current_status = read_rebuild_status(dependencies.runtime_root)
        current_state = str(current_status.get("status", "idle"))
        if current_state in REBUILD_ACTIVE_STATUSES:
            return {
                "status": "ok",
                "accepted": False,
                "already_running": True,
                "rebuild": current_status,
            }

        project_override = (
            str(payload.get("project"))
            if payload.get("project") is not None
            else None
        )
        scope = resolve_rebuild_scope(
            ctx.config,
            ctx.documents_roots,
            project_override,
        )
        request_id = uuid4().hex
        submission = submit_rebuild_request(
            project_override,
            request_id=request_id,
        )
        if submission.status == "already_pending":
            return {
                "status": "ok",
                "accepted": False,
                "already_running": True,
                "rebuild": read_rebuild_status(dependencies.runtime_root),
            }
        if not submission.queue_available:
            return {
                "status": "error",
                "error": "rebuild_queue_unavailable",
                "details": "Daemon rebuild queue is unavailable.",
            }
        if submission.should_retry_later:
            return {
                "status": "ok",
                "accepted": False,
                "already_running": False,
                "retry_later": True,
                "details": "Daemon rebuild writer is busy. Retry shortly.",
            }

        queued_status = submit_rebuild_status(
            dependencies.runtime_root,
            request_id=request_id,
            scope=scope,
        )
        return {
            "status": "ok",
            "accepted": submission.accepted_by_queue,
            "already_running": False,
            "rebuild": queued_status,
        }
        return None


def _handle_internal_request(
    path: str,
    coordinator: _RouterCoordinator,
) -> dict[str, object] | None:
    if path != "/internal/shutdown":
        return None
    coordinator.request_shutdown()
    return {"status": "ok", "lifecycle": coordinator.state.value}


async def _handle_search_request(
    path: str,
    payload: dict[str, object],
    ctx: _RouterContext,
    coordinator: _RouterCoordinator,
) -> dict[str, object] | None:
    if path == "/api/search/query":
        cold_start_response = _get_cold_start_search_response(
            ctx,
            coordinator,
            query=str(payload.get("query", "")),
        )
        if cold_start_response is not None:
            return cold_start_response
        await ctx.ensure_fresh_indices()
        ctx.schedule_freshness_refresh()
        query_text = str(payload.get("query", ""))
        top_n = _as_int(payload.get("top_n"), 5)
        project_filter_payload = payload.get("project_filter", [])
        project_filter = (
            [str(item) for item in project_filter_payload if isinstance(item, str)]
            if isinstance(project_filter_payload, list)
            else []
        )
        source_filter_payload = payload.get("source_filter", [])
        source_filter = (
            [str(item) for item in source_filter_payload if isinstance(item, str)]
            if isinstance(source_filter_payload, list)
            else []
        )
        search_use_case = getattr(ctx, "search_use_case", None)
        if search_use_case is None:
            search_use_case = getattr(ctx.orchestrator, "search_use_case", None)
        query_execution_stats: dict[str, object] = {}
        if search_use_case is not None:
            execution = await search_use_case.execute(
                SearchQuery(
                    query=query_text,
                    top_n=top_n,
                    project_filter=tuple(project_filter),
                    source_filter=tuple(source_filter),
                    project_context=(
                        str(payload.get("project_context"))
                        if payload.get("project_context") is not None
                        else None
                    ),
                )
            )
            results = execution.results
            compression_stats = execution.compression_stats
            strategy_stats = execution.strategy_stats
            query_execution_stats = execution.query_execution_stats
        else:
            top_k = max(20, top_n * 4)
            if project_filter:
                top_k = max(top_k, top_n * 10)
            results, compression_stats, strategy_stats = await ctx.orchestrator.query(
                query_text,
                top_k=top_k,
                top_n=top_n,
                project_filter=project_filter,
                source_filter=source_filter,
                project_context=(
                    str(payload.get("project_context"))
                    if payload.get("project_context") is not None
                    else None
                ),
            )
            results = [
                result if isinstance(result, ChunkResult) else ChunkResult.from_domain(result)
                for result in results
            ]
            query_execution_stats = ctx.orchestrator.last_query_execution_stats or {}
        await ctx.orchestrator.drain_reindex()
        return {
            "query": query_text,
            "results": [result.to_dict() for result in results],
            "compression_stats": compression_stats.to_dict(),
            "strategy_stats": strategy_stats.to_dict(),
            "query_execution_stats": query_execution_stats,
        }
    if path == "/api/search/git-history":
        if not ctx.git_indexing_enabled:
            return {"status": "error", "error": "git_indexing_unavailable"}

        cold_start_response = _get_cold_start_search_response(
            ctx,
            coordinator,
            query=str(payload.get("query", "")),
            include_git_metadata=True,
        )
        if cold_start_response is not None:
            return cold_start_response

        await ctx.ensure_fresh_indices()
        ctx.schedule_freshness_refresh()

        query_text = str(payload.get("query", ""))
        top_n = _as_int(payload.get("top_n"), 5)
        project_filter_payload = payload.get("project_filter", [])
        project_filter = (
            [str(item) for item in project_filter_payload if isinstance(item, str)]
            if isinstance(project_filter_payload, list)
            else []
        )
        project_context = (
            str(payload.get("project_context"))
            if payload.get("project_context") is not None
            else None
        )
        after_timestamp = _as_int(payload.get("after_timestamp")) if payload.get("after_timestamp") is not None else None
        before_timestamp = _as_int(payload.get("before_timestamp")) if payload.get("before_timestamp") is not None else None
        files_glob = str(payload["files_glob"]) if payload.get("files_glob") else None

        overfetch_multiplier = 10 if (files_glob or after_timestamp or before_timestamp) else 4
        results, _, _ = await ctx.orchestrator.query(
            query_text,
            top_k=max(20, top_n * overfetch_multiplier),
            top_n=top_n * overfetch_multiplier,
            project_filter=project_filter,
            project_context=project_context,
            source_filter=["git_commit"],
        )
        results = [
            result if isinstance(result, ChunkResult) else ChunkResult.from_domain(result)
            for result in results
        ]
        commits = _filter_git_history_results(
            results, files_glob, after_timestamp, before_timestamp
        )[:top_n]

        return {
            "query": query_text,
            "total_commits_indexed": ctx.get_total_git_commits_indexed(),
            "results": [_git_history_result_to_dict(result) for result in commits],
        }
    return None

def build_daemon_request_handler(
    dependencies: DaemonRequestRouterDependencies,
) -> Callable[[str, dict[str, object]], Coroutine[Any, Any, dict[str, object]]]:
    async def _handle_daemon_request(
        path: str,
        payload: dict[str, object],
    ) -> dict[str, object]:
        ctx = cast(_RouterContext, dependencies.ctx)
        coordinator = cast(_RouterCoordinator, dependencies.coordinator)

        mcp_response = await _handle_mcp_request(path, payload, ctx, coordinator)
        if mcp_response is not None:
            return mcp_response
        if path == "/api/index/records":
            return await _submit_record_batch(dependencies, payload)
        admin_response = await _handle_admin_request(
            dependencies, path, payload, ctx, coordinator
        )
        if admin_response is not None:
            return admin_response
        internal_response = _handle_internal_request(path, coordinator)
        if internal_response is not None:
            return internal_response
        search_response = await _handle_search_request(
            path, payload, ctx, coordinator
        )
        if search_response is not None:
            return search_response
        return {"status": "error", "error": "unknown_request_path", "path": path}

    return _handle_daemon_request
