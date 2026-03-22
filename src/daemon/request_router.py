from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from src.daemon.mcp_requests import build_mcp_tools_payload, handle_mcp_tool_call
from src.indexing.rebuild_service import (
    REBUILD_ACTIVE_STATUSES,
    read_rebuild_status,
    resolve_rebuild_scope,
    submit_rebuild_status,
    write_rebuild_status,
)
from src.indexing.tasks import submit_rebuild_request


type BuildAdminOverviewPayload = Callable[[object, Path, bool, int | None, str], dict[str, object]]
type BuildIndexStatsPayload = Callable[[object], dict[str, object]]
type BuildQueueStatusPayload = Callable[[Path, bool, int | None], dict[str, object]]


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
        payload["total_commits_indexed"] = (
            ctx.commit_indexer.get_total_commits() if ctx.commit_indexer is not None else 0
        )
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
            if ctx.commit_indexer is None:
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

            from src.git.commit_search import search_git_history

            response = search_git_history(
                commit_indexer=ctx.commit_indexer,
                query=str(payload.get("query", "")),
                top_n=int(payload.get("top_n", 5)),
                files_glob=str(payload["files_glob"]) if payload.get("files_glob") else None,
                after_timestamp=int(payload["after_timestamp"]) if payload.get("after_timestamp") is not None else None,
                before_timestamp=int(payload["before_timestamp"]) if payload.get("before_timestamp") is not None else None,
                project_filter=(
                    [str(item) for item in payload.get("project_filter", []) if isinstance(item, str)]
                    if isinstance(payload.get("project_filter", []), list)
                    else []
                ),
                project_context=(
                    str(payload.get("project_context"))
                    if payload.get("project_context") is not None
                    else None
                ),
                config=ctx.config,
            )
            return {
                "query": response.query,
                "total_commits_indexed": response.total_commits_indexed,
                "results": [
                    {
                        "hash": result.hash,
                        "title": result.title,
                        "author": result.author,
                        "committer": result.committer,
                        "timestamp": result.timestamp,
                        "message": result.message,
                        "files_changed": result.files_changed,
                        "delta_truncated": result.delta_truncated,
                        "score": result.score,
                        "repo_path": result.repo_path,
                        "project_id": result.project_id,
                    }
                    for result in response.results
                ],
            }
        return {"status": "error", "error": "unknown_request_path", "path": path}

    return _handle_daemon_request