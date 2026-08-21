"""Payload builders for daemon admin/index/queue status responses."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from mcp_markdown_ragdocs.coordination.queue import QueueRuntime
from mcp_markdown_ragdocs.daemon.queue_status import get_queue_stats
from mcp_markdown_ragdocs.daemon.producer import (
    producer_diagnostics,
    read_producer_metadata,
)
from mcp_markdown_ragdocs.daemon.storage_diagnostics import sqlite_storage_diagnostics
from mcp_markdown_ragdocs.indexing.git_refresh_state import list_progress
from mcp_markdown_ragdocs.indexing.reindex import reindex_status_payload

if TYPE_CHECKING:
    from mcp_markdown_ragdocs.context import ApplicationContext
    from mcp_markdown_ragdocs.daemon import RuntimePaths

logger = logging.getLogger(__name__)


def _as_int(value: object) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def _resolve_stats_file_path(
    file_path: str | None,
    *,
    common_root: Path,
) -> Path | None:
    if not file_path:
        return None

    candidate = Path(file_path).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()

    return (common_root / candidate).resolve()


def _match_documents_root(
    file_path: Path,
    documents_roots: list[Path],
) -> int | None:
    for index, root in enumerate(documents_roots):
        try:
            file_path.relative_to(root)
            return index
        except ValueError:
            continue

    return None


def _build_per_root_index_rows(
    ctx: ApplicationContext,
    *,
    discovered_files: list[str],
    common_root: Path,
    include_indexed_estimates: bool,
    indexed_descriptions: list[dict[str, object]] | None = None,
) -> tuple[list[dict[str, object]], int, int]:
    documents_roots = [root.resolve() for root in ctx.documents_roots]
    rows = [
        {
            "root_path": str(root),
            "discovered_files": 0,
            "indexed_documents_estimate": 0,
            "indexed_chunks_estimate": 0,
            "remaining_estimate": 0,
        }
        for root in documents_roots
    ]

    for discovered_file in discovered_files:
        resolved_path = Path(discovered_file).expanduser().resolve()
        root_index = _match_documents_root(resolved_path, documents_roots)
        if root_index is None:
            continue
        rows[root_index]["discovered_files"] += 1

    unattributed_indexed_documents = 0
    unattributed_indexed_chunks = 0
    if include_indexed_estimates:
        descriptions = (
            indexed_descriptions
            if indexed_descriptions is not None
            else ctx.index_manager.describe_documents()
        )
        for description in descriptions:
            raw_file_path = description.get("file_path")
            resolved_path = _resolve_stats_file_path(
                raw_file_path if isinstance(raw_file_path, str) else None,
                common_root=common_root,
            )
            chunk_count = _as_int(description.get("chunk_count"))
            if resolved_path is None:
                unattributed_indexed_documents += 1
                unattributed_indexed_chunks += chunk_count
                continue

            root_index = _match_documents_root(resolved_path, documents_roots)
            if root_index is None:
                unattributed_indexed_documents += 1
                unattributed_indexed_chunks += chunk_count
                continue

            rows[root_index]["indexed_documents_estimate"] += 1
            rows[root_index]["indexed_chunks_estimate"] += chunk_count

    for row in rows:
        row["remaining_estimate"] = max(
            _as_int(row["discovered_files"]) - _as_int(row["indexed_documents_estimate"]),
            0,
        )

    return (
        cast(list[dict[str, object]], rows),
        unattributed_indexed_documents,
        unattributed_indexed_chunks,
    )


def _build_index_stats_payload(ctx: Any) -> dict[str, object]:
    manifest_path = ctx.index_path / "index.manifest.json"
    manifest_exists = manifest_path.exists()
    persisted_index_exists = manifest_exists or (ctx.index_path / "index.db").exists()
    if persisted_index_exists:
        try:
            ctx.index_manager.load()
        except TimeoutError as exc:
            if "Failed to acquire shared lock" not in str(exc):
                raise
            logger.warning(
                "Index stats refresh skipped after shared-lock timeout; using current in-memory snapshot",
                exc_info=True,
            )

    docs_root = Path(ctx.config.indexing.documents_path).resolve()
    discovered_files = ctx.discover_files() if docs_root.exists() else []
    repo_count = len(ctx.discover_git_repositories())
    git_commit_count = ctx.get_total_git_commits_indexed()

    indexed_descriptions: list[dict[str, object]] = []
    indexed_documents = 0
    indexed_chunks = 0
    if persisted_index_exists:
        indexed_descriptions = ctx.index_manager.describe_documents()
        indexed_documents = len(indexed_descriptions)
        indexed_chunks = sum(
            _as_int(description.get("chunk_count"))
            for description in indexed_descriptions
        )

    per_root_rows, unattributed_indexed_documents, unattributed_indexed_chunks = (
        _build_per_root_index_rows(
            ctx,
            discovered_files=discovered_files,
            common_root=docs_root,
            include_indexed_estimates=persisted_index_exists,
            indexed_descriptions=indexed_descriptions,
        )
    )
    remaining_estimate = max(len(discovered_files) - indexed_documents, 0)

    return {
        "documents_path": str(docs_root),
        "documents_common_root": str(docs_root),
        "documents_path_kind": "common_root",
        "documents_roots": [str(root) for root in ctx.documents_roots],
        "index_path": str(ctx.index_path),
        "index_db_path": str(ctx.index_path / "index.db"),
        "storage": sqlite_storage_diagnostics(ctx.index_path / "index.db"),
        "manifest_path": str(manifest_path),
        "manifest_exists": manifest_exists,
        "indexed_documents": indexed_documents,
        "indexed_chunks": indexed_chunks,
        "discovered_files": len(discovered_files),
        "remaining_estimate": remaining_estimate,
        "per_root": per_root_rows,
        "per_root_counts_are_estimates": True,
        "unattributed_indexed_documents": unattributed_indexed_documents,
        "unattributed_indexed_chunks": unattributed_indexed_chunks,
        "git_commits": git_commit_count,
        "git_repositories": repo_count,
        "index_state": ctx.get_index_state().to_dict(),
        "watcher_stats": ctx.watcher.get_stats().to_dict() if ctx.watcher else None,
    }


def _build_queue_status_payload(
    *,
    queue_runtime: QueueRuntime,
    worker_running: bool,
    backpressure_limit: int | None = None,
) -> dict[str, object]:
    stats = get_queue_stats(
        queue_runtime.huey,
        worker_running=worker_running,
        backpressure_limit=backpressure_limit,
    )
    payload = stats.to_dict()
    payload["queue_db_path"] = str(queue_runtime.db_path)
    payload["git_refresh_progress"] = list_progress(queue_runtime.db_path.parent)
    return payload


def _build_admin_overview_payload(
    ctx: ApplicationContext,
    runtime_paths: RuntimePaths,
    queue_runtime: QueueRuntime,
    worker_running: bool,
    worker_pid: int | None,
    lifecycle: str,
) -> dict[str, object]:
    index_payload = _build_index_stats_payload(ctx)
    task_payload = _build_queue_status_payload(
        queue_runtime=queue_runtime,
        worker_running=worker_running,
        backpressure_limit=ctx.config.indexing.task_backpressure_limit,
    )
    watcher_stats = ctx.watcher.get_stats().to_dict() if ctx.watcher else None
    producer_path = runtime_paths.producer_metadata_path or (
        runtime_paths.root / "producer.json"
    )
    producer_payload = producer_diagnostics(read_producer_metadata(producer_path))
    return {
        "status": "ok",
        "pid": os.getpid(),
        "lifecycle": lifecycle,
        "daemon_scope": "global",
        "project_context_mode": "request_only",
        "configured_root_count": len(ctx.documents_roots),
        "documents_roots": [str(root) for root in ctx.documents_roots],
        "worker_health": "healthy" if worker_running else "dead",
        "worker_pid": worker_pid,
        "socket_path": str(runtime_paths.socket_path),
        "endpoint": f"ipc://{runtime_paths.socket_path}",
        "index_db_path": str(runtime_paths.index_db_path),
        "storage": index_payload["storage"],
        "queue_db_path": str(runtime_paths.queue_db_path),
        "indexed_documents": index_payload["indexed_documents"],
        "indexed_chunks": index_payload["indexed_chunks"],
        "git_commits": index_payload["git_commits"],
        "git_repositories": index_payload["git_repositories"],
        "pending_count": task_payload["pending_count"],
        "scheduled_count": task_payload["scheduled_count"],
        "running_count": task_payload["running_count"],
        "failed_count": task_payload["failed_count"],
        "worker_running": task_payload["worker_running"],
        "queue_stats": task_payload,
        "git_refresh_progress": task_payload.get("git_refresh_progress", []),
        "watcher_stats": watcher_stats,
        **producer_payload,
        "index_state": ctx.get_index_state().to_dict(),
        "reindex": reindex_status_payload(runtime_paths.root, ctx.index_path),
    }
