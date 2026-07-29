from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast
from uuid import uuid4

from searchkernel.daemon.mcp_requests import build_mcp_tools_payload, handle_mcp_tool_call
from searchkernel.daemon.record_rpc import RecordSerializationError, deserialize_record
from searchkernel.domain import Record
from searchkernel.indexing.rebuild_service import (
    REBUILD_ACTIVE_STATUSES,
    read_rebuild_status,
    resolve_rebuild_scope,
    submit_rebuild_status,
    write_rebuild_status,
)
from searchkernel.indexing.tasks import submit_rebuild_request


type BuildAdminOverviewPayload = Callable[[object, Path, bool, int | None, str], dict[str, object]]
type BuildIndexStatsPayload = Callable[[object], dict[str, object]]
type BuildQueueStatusPayload = Callable[[Path, bool, int | None], dict[str, object]]


class _RecordIndexManager(Protocol):
    def index_record(self, record: Record) -> None: ...


class _RecordIndexContext(Protocol):
    index_manager: _RecordIndexManager


def _index_records(ctx: _RecordIndexContext, payload: dict[str, object]) -> dict[str, object]:
    raw_records = payload.get("records")
    if not isinstance(raw_records, list):
        return {
            "status": "error",
            "error": "records_must_be_list",
            "details": "The request payload must contain a records list.",
        }

    records: list[Record] = []
    for index, raw_record in enumerate(raw_records):
        try:
            records.append(deserialize_record(raw_record))
        except RecordSerializationError as exc:
            return {
                "status": "error",
                "error": "invalid_record",
                "record_index": index,
                "details": str(exc),
            }

    for index, record in enumerate(records):
        try:
            ctx.index_manager.index_record(record)
        except Exception as exc:
            return {
                "status": "error",
                "error": "record_indexing_failed",
                "record_index": index,
                "indexed_count": index,
                "details": str(exc),
            }

    return {"status": "ok", "indexed_count": len(records)}


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
    ctx,
    coordinator,
    *,
    query: str,
    include_git_metadata: bool = False,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "status": "initializing",
        "message": "Search indices are still initializing. Retry shortly.",
        "query": query,
        "results": [],
        "lifecycle": coordinator.state.value,
        "daemon_scope": "global",
        "project_context_mode": "request_only",
        "configured_root_count": len(ctx.documents_roots),
        "index_state": ctx.get_index_state().to_dict(),
    }
    if include_git_metadata:
        payload["total_commits_indexed"] = ctx.get_total_git_commits_indexed()
    else:
        payload["compression_stats"] = {}
        payload["strategy_stats"] = {}
    return payload


def _build_unavailable_search_payload(
    ctx,
    coordinator,
) -> dict[str, object]:
    index_state = ctx.get_index_state()
    return {
        "status": "error",
        "error": "index_initialization_failed",
        "details": index_state.last_error or "Search indices are not queryable.",
        "lifecycle": coordinator.state.value,
        "daemon_scope": "global",
        "project_context_mode": "request_only",
        "configured_root_count": len(ctx.documents_roots),
        "index_state": index_state.to_dict(),
    }


def _get_cold_start_search_response(
    ctx,
    coordinator,
    *,
    query: str,
    include_git_metadata: bool = False,
) -> dict[str, object] | None:
    if ctx.is_ready():
        return None

    index_state = ctx.get_index_state()
    if index_state.status in {"failed", "partial"}:
        return _build_unavailable_search_payload(ctx, coordinator)

    return _build_initializing_search_payload(
        ctx,
        coordinator,
        query=query,
        include_git_metadata=include_git_metadata,
    )


def build_daemon_request_handler(
    dependencies: DaemonRequestRouterDependencies,
) -> Callable[[str, dict[str, object]], Awaitable[dict[str, object]]]:
    async def _handle_daemon_request(
        path: str,
        payload: dict[str, object],
    ) -> dict[str, object]:
        ctx = dependencies.ctx
        coordinator = dependencies.coordinator

        if path == "/api/mcp/tools":
            return build_mcp_tools_payload()
        if path == "/api/mcp/tool":
            return await handle_mcp_tool_call(
                ctx_getter=lambda: ctx,
                coordinator=coordinator,
                payload=payload,
            )
        if path == "/api/index/records":
            return _index_records(cast(_RecordIndexContext, ctx), payload)
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
                dependencies.queue_db_path,
                dependencies.get_worker_running(),
                ctx.config.indexing.task_backpressure_limit,
            )
        if path == "/api/admin/rebuild/status":
            return read_rebuild_status(dependencies.runtime_root)
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
            queued_status = submit_rebuild_status(
                dependencies.runtime_root,
                request_id=request_id,
                scope=scope,
            )
            submission = submit_rebuild_request(
                project_override,
                request_id=request_id,
            )
            if not submission.queue_available:
                write_rebuild_status(dependencies.runtime_root, {"status": "idle"})
                return {
                    "status": "error",
                    "error": "rebuild_queue_unavailable",
                    "details": "Daemon rebuild queue is unavailable.",
                }
            if submission.should_retry_later:
                write_rebuild_status(dependencies.runtime_root, {"status": "idle"})
                return {
                    "status": "error",
                    "error": "rebuild_queue_backpressured",
                    "details": "Daemon rebuild queue is backpressured. Retry shortly.",
                }

            return {
                "status": "ok",
                "accepted": submission.accepted_by_queue,
                "already_running": False,
                "rebuild": queued_status,
            }
        if path == "/internal/shutdown":
            coordinator.request_shutdown()
            return {"status": "ok", "lifecycle": coordinator.state.value}
        if path == "/api/search/query":
            cold_start_response = _get_cold_start_search_response(
                ctx,
                coordinator,
                query=str(payload.get("query", "")),
            )
            if cold_start_response is not None:
                return cold_start_response
            ctx.schedule_freshness_refresh()
            query_text = str(payload.get("query", ""))
            top_n = int(payload.get("top_n", 5))
            top_k = max(20, top_n * 4)
            project_filter_payload = payload.get("project_filter", [])
            project_filter = (
                [str(item) for item in project_filter_payload if isinstance(item, str)]
                if isinstance(project_filter_payload, list)
                else []
            )
            if project_filter:
                top_k = max(top_k, top_n * 10)
            results, compression_stats, strategy_stats = await ctx.orchestrator.query(
                query_text,
                top_k=top_k,
                top_n=top_n,
                project_filter=project_filter,
                project_context=(
                    str(payload.get("project_context"))
                    if payload.get("project_context") is not None
                    else None
                ),
            )
            await ctx.orchestrator.drain_reindex()
            return {
                "query": query_text,
                "results": [result.to_dict() for result in results],
                "compression_stats": compression_stats.to_dict(),
                "strategy_stats": strategy_stats.to_dict(),
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

            ctx.schedule_freshness_refresh()

            query_text = str(payload.get("query", ""))
            top_n = int(payload.get("top_n", 5))
            project_filter = (
                [str(item) for item in payload.get("project_filter", []) if isinstance(item, str)]
                if isinstance(payload.get("project_filter", []), list)
                else []
            )
            project_context = (
                str(payload.get("project_context"))
                if payload.get("project_context") is not None
                else None
            )
            after_timestamp = (
                int(payload["after_timestamp"]) if payload.get("after_timestamp") is not None else None
            )
            before_timestamp = (
                int(payload["before_timestamp"]) if payload.get("before_timestamp") is not None else None
            )
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
            commits = _filter_git_history_results(
                results, files_glob, after_timestamp, before_timestamp
            )[:top_n]

            return {
                "query": query_text,
                "total_commits_indexed": ctx.get_total_git_commits_indexed(),
                "results": [_git_history_result_to_dict(result) for result in commits],
            }
        return {"status": "error", "error": "unknown_request_path", "path": path}

    return _handle_daemon_request
