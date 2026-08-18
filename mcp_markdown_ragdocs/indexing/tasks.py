"""Huey task definitions for indexing operations."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import threading
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, cast

from searchkernel.api import (
    TaskBatchSubmissionResult,
    TaskSubmissionResult,
    build_indexed_files_map,
    has_incomplete_bootstrap_checkpoint,
    load_manifest,
    mark_bootstrap_file_completed,
    mark_bootstrap_files_completed,
    save_manifest,
)

from mcp_markdown_ragdocs.coordination.task_leases import TaskLeasePort
from mcp_markdown_ragdocs.coordination.task_submission import (
    coalesce_pending_first_args,
    get_pending_task_first_args,
    get_pending_task_values,
    is_backpressured,
    submit_coalesced_batch_task,
)
from mcp_markdown_ragdocs.coordination.task_submission import (
    get_pending_task_count as get_shared_pending_task_count,
)
from mcp_markdown_ragdocs.coordination.work_intents import WorkIntent, WorkIntentPort
from mcp_markdown_ragdocs.git.repository import get_git_ref_signature
from mcp_markdown_ragdocs.indexing.git_refresh_state import (
    get_cursor,
    get_head,
    save_cursor,
    save_head,
    save_progress,
)
from mcp_markdown_ragdocs.indexing.progressive import (
    ProgressiveIndexManager,
    run_progressive_bootstrap,
)
from mcp_markdown_ragdocs.indexing.rebuild_service import (
    run_rebuild,
    write_rebuild_status,
)
from mcp_markdown_ragdocs.indexing.reindex import (
    run_reindex_operation,
    write_reindex_status,
)
from mcp_markdown_ragdocs.indexing import task_intents
from mcp_markdown_ragdocs.indexing.task_registration import register_huey_tasks
from mcp_markdown_ragdocs.indexing.task_writer import (
    WRITER_HEARTBEAT_INTERVAL_SECONDS,  # noqa: F401 - compatibility export
    WRITER_LEASE_TIMEOUT_SECONDS,  # noqa: F401 - compatibility export
    run_as_writer,
    writer_is_active,
    writer_owned_task,
)
from mcp_markdown_ragdocs.gdrive.tasks import (
    GDriveTaskManager,
    build_gdrive_task_runtime,
    register_gdrive_tasks,
)

if TYPE_CHECKING:
    from huey import SqliteHuey
    from searchkernel.api import Record

logger = logging.getLogger(__name__)

RECORD_BATCH_TASK_PRIORITY = 100
GIT_REFRESH_TASK_PRIORITY = -10
REINDEX_TASK_PRIORITY = 200
GIT_REFRESH_BATCH_SIZE = 200
DOCUMENT_TASK_BATCH_SIZE = 32

__all__ = [
    "TaskBatchSubmissionResult",
    "TaskSubmissionResult",
]


class IndexManagerLike(Protocol):
    """Structural type for objects that can index/remove documents."""

    ingestor: Any

    def index_document(self, file_path: str, force: bool = False) -> bool: ...
    def index_documents(
        self,
        file_paths: list[str],
        force: bool = False,
        persist: bool = False,
    ) -> None: ...
    def remove_document(self, doc_id: str) -> None: ...
    def remove_documents(
        self,
        doc_ids: list[str],
        persist: bool = False,
    ) -> None: ...
    def persist(self) -> None: ...
    def index_record(self, record: Record) -> Any: ...


# Module-level references set during initialization
_huey: SqliteHuey | None = None
_index_manager: IndexManagerLike | None = None
_task_backpressure_limit: int = 100
_bootstrap_index_path: Path | None = None
_bootstrap_documents_roots: list[Path] = []
_schedule_vocabulary_catch_up: Callable[[], bool] | None = None
_task_lease_store: TaskLeasePort | None = None
_work_intent_store: WorkIntentPort | None = None
_git_refresh_in_flight: set[str] = set()
_git_refresh_pending: set[str] = set()
_git_refresh_deferred: set[str] = set()
_git_refresh_lock = threading.Lock()

# Task references (set after register_tasks is called)
index_document_task: Any = None
index_documents_batch_task: Any = None
index_records_batch_task: Any = None
remove_document_task: Any = None
remove_documents_batch_task: Any = None
refresh_git_repository_task: Any = None
rebuild_index_task: Any = None
reindex_model_task: Any = None
gdrive_inventory_task: Any = None
gdrive_startup_task: Any = None
gdrive_changes_task: Any = None
gdrive_retry_task: Any = None
gdrive_backfill_task: Any = None
gdrive_lease_task: Any = None
gdrive_watch_task: Any = None
gdrive_health_task: Any = None


def _writer_lease_store() -> TaskLeasePort | None:
    return _task_lease_store


def _intent_store() -> WorkIntentPort | None:
    return _work_intent_store


def _canonical_document_identity(document_id: str) -> str:
    path = Path(document_id).expanduser()
    if path.is_absolute() or "/" in document_id or "\\" in document_id:
        return str(path.resolve())
    return document_id


def _unique_document_paths(file_paths: list[str]) -> list[str]:
    unique_paths: list[str] = []
    seen: set[str] = set()
    for file_path in file_paths:
        identity = _canonical_document_identity(file_path)
        if identity not in seen:
            seen.add(identity)
            unique_paths.append(file_path)
    return unique_paths


def _intent_claim(
    operation: str,
    canonical_key: str,
    payload: dict[str, object],
    *,
    force_reopen: bool = False,
) -> tuple[WorkIntent, str] | None:
    return task_intents._intent_claim(
        _intent_store, operation, canonical_key, payload, force_reopen=force_reopen
    )


def _intent_claim_batch(
    operation: str,
    items: list[tuple[str, dict[str, object]]],
    *,
    force_reopen: bool = False,
) -> tuple[list[tuple[str, tuple[str, str]]], int]:
    return task_intents._intent_claim_batch(
        _intent_store, operation, items, force_reopen=force_reopen
    )


def _release_intent(intent_id: str, claim_token: str) -> None:
    task_intents._release_intent(_intent_store, intent_id, claim_token)


def _intent_task(operation: str):
    return task_intents._intent_task(_intent_store, operation)


def _batch_result(outcomes: list[bool], *, error: str) -> dict[str, object]:
    return {
        "status": "ok" if all(outcomes) else "error",
        "error": None if all(outcomes) else error,
        "outcomes": outcomes,
    }


def _failed_file_details(manager: Any, file_path: str) -> list[dict[str, str]]:
    get_failed_files = getattr(manager, "get_failed_files", None)
    if not callable(get_failed_files):
        return []


    try:
        failed_files = get_failed_files()
        if not isinstance(failed_files, list):
            return []
        return [
            {"path": str(item.get("path")), "error": str(item.get("error"))}
            for item in failed_files
            if isinstance(item, dict) and item.get("path") == file_path
        ][-3:]
    except Exception:  # noqa: BLE001 - diagnostics must not alter task behavior
        return []


def _persist_document_manifest(
    *,
    indexed_paths: list[str] | None = None,
    removed_doc_ids: list[str] | None = None,
) -> None:
    if _bootstrap_index_path is None or not _bootstrap_documents_roots:
        return

    manifest = load_manifest(_bootstrap_index_path)
    if manifest is None:
        return

    indexed_files = dict(manifest.indexed_files or {})
    indexed_files.update(
        build_indexed_files_map(
            indexed_paths or [],
            _bootstrap_documents_roots[0],
            _bootstrap_documents_roots,
        )
    )
    for doc_id in removed_doc_ids or []:
        indexed_files.pop(doc_id, None)

    manifest.indexed_files = indexed_files
    save_manifest(_bootstrap_index_path, manifest)


def _writer_is_active() -> bool:
    return writer_is_active(_writer_lease_store)


def _run_as_writer(
    operation: Callable[[], Any],
    *,
    operation_name: str = "index write",
    operation_args: tuple[Any, ...] = (),
    owner_token: str | None = None,
    busy_result: Any,
    on_busy: Callable[[], None] | None = None,
) -> Any:
    return run_as_writer(
        _writer_lease_store,
        operation,
        operation_name=operation_name,
        operation_args=operation_args,
        owner_token=owner_token,
        busy_result=busy_result,
        on_busy=on_busy,
        on_released=_flush_deferred_git_refreshes,
    )


def _writer_owned_task(
    *,
    operation: str | None = None,
    busy_result: Any,
    on_busy: Callable[..., None] | None = None,
):
    return writer_owned_task(
        _writer_lease_store,
        operation=operation,
        busy_result=busy_result,
        on_busy=on_busy,
        on_released=_flush_deferred_git_refreshes,
    )


class _RejectedRecordBatch:
    def get(self, *, blocking: bool, timeout: float) -> dict[str, object]:
        return {
            "status": "error",
            "error": "index_writer_busy",
            "details": "Index writer is owned by a rebuild. Retry shortly.",
        }


def _git_refresh_key(git_dir: str | Path) -> str:
    return str(Path(git_dir).resolve())


def _embedding_cache_metrics(manager: IndexManagerLike) -> dict[str, int]:
    cache = getattr(manager, "_embedding_cache", None)
    metrics = getattr(cache, "metrics", None)
    if metrics is None:
        return {}
    names = {
        "hits": "embedding_cache_hits",
        "misses": "embedding_cache_misses",
        "writes": "embedding_writes",
        "invalidations": "embedding_invalidations",
    }
    return {
        output_name: int(value)
        for metric_name, output_name in names.items()
        if isinstance(value := getattr(metrics, metric_name, None), int)
        and not isinstance(value, bool)
    }


def _refresh_progress_rate(processed: int, started_at: datetime | None) -> float:
    if started_at is None:
        return 0.0
    elapsed = max((datetime.now(UTC) - started_at).total_seconds(), 0.001)
    return processed / elapsed


def _git_commit_id_from_result(result: object) -> str | None:
    """Extract the logical commit ID from a chunk or ingestion result."""

    source_id = getattr(result, "source_id", None)
    if isinstance(source_id, str):
        parts = source_id.split(":")
        if len(parts) >= 2 and parts[0] == "git":
            return ":".join(parts[:2])
    metadata = getattr(result, "metadata", None)
    if isinstance(metadata, dict):
        commit_id = metadata.get("commit_id")
        if isinstance(commit_id, str):
            return commit_id
    return None


def _newest_commit_from_ref_signature(signature: str | None) -> str | None:
    if not signature:
        return None
    head = signature.splitlines()[0].strip()
    return head or None


def _save_git_refresh_progress(
    git_dir: Path,
    *,
    state: str,
    started_at: datetime | None = None,
    updated_at: datetime | None = None,
    metrics: dict[str, int] | None = None,
    **fields: object,
) -> None:
    if _bootstrap_index_path is None:
        return
    timestamp = updated_at or datetime.now(UTC)
    progress: dict[str, object] = {
        "state": state,
        "updated_at": timestamp.isoformat(),
    }
    if started_at is not None:
        progress["started_at"] = started_at.isoformat()
    if "processed_count" in fields and isinstance(fields["processed_count"], int):
        progress["rate_per_second"] = _refresh_progress_rate(
            fields["processed_count"], started_at
        )
    progress.update(fields)
    if metrics:
        progress.update(metrics)
    save_progress(_bootstrap_index_path, git_dir, progress)


def _defer_git_refresh(git_dir: str) -> None:
    repo_key = _git_refresh_key(git_dir)
    with _git_refresh_lock:
        _git_refresh_pending.add(repo_key)
        _git_refresh_deferred.add(repo_key)
    logger.info("Deferred git refresh until index writer release: %s", git_dir)


def _flush_deferred_git_refreshes() -> None:
    with _git_refresh_lock:
        deferred = list(_git_refresh_deferred)
        _git_refresh_deferred.difference_update(deferred)
        _git_refresh_pending.difference_update(deferred)

    for git_dir in deferred:
        submission = submit_refresh_git_request(git_dir)
        if submission.should_retry_later or not submission.queue_available:
            with _git_refresh_lock:
                _git_refresh_pending.add(git_dir)
                _git_refresh_deferred.add(git_dir)
            logger.info(
                "Keeping deferred git refresh for a later retry: %s",
                git_dir,
            )


def _index_document(file_path: str, force: bool = False) -> bool:
    """Index or re-index a single document."""
    if _index_manager is None:
        logger.error("IndexManager not available for task execution")
        return False
    manager = _index_manager
    logger.info("Index task started: path=%s force=%s", file_path, force)

    def _operation() -> bool:
        try:
            manager_result = manager.index_document(file_path, force=force)
            logger.info(
                "Index task manager result: path=%s force=%s result=%s failed_files=%s",
                file_path,
                force,
                manager_result,
                _failed_file_details(manager, file_path) if not manager_result else [],
            )
            manager.persist()
            if manager_result:
                _persist_document_manifest(indexed_paths=[file_path])
            if _bootstrap_index_path is not None and _bootstrap_documents_roots:
                mark_bootstrap_file_completed(
                    _bootstrap_index_path,
                    _bootstrap_documents_roots,
                    file_path,
                )
            logger.info("Task completed: indexed %s", file_path)
            return True
        except Exception:
            logger.exception("Task failed: index %s", file_path)
            return False

    return _run_as_writer(
        _operation,
        operation_name="index_document",
        operation_args=(file_path,),
        busy_result=False,
    )

def _index_documents_batch(
    file_paths: list[str],
    force: bool = False,
    progressive: bool = False,
) -> bool:
    """Index a burst of documents and persist once after the batch."""
    if _index_manager is None:
        logger.error("IndexManager not available for batch task execution")
        return False
    manager = _index_manager

    unique_file_paths = list(dict.fromkeys(file_paths))
    if not unique_file_paths:
        return True

    progressive_index = getattr(
        _index_manager,
        "prepare_progressive_document",
        None,
    )
    canonical_index = hasattr(_index_manager, "kernel")
    if (
        progressive
        and not force
        and (progressive_index is not None or canonical_index)
        and _bootstrap_index_path is not None
        and _bootstrap_documents_roots
        and has_incomplete_bootstrap_checkpoint(_bootstrap_index_path)
    ):
        def _progressive_operation() -> bool | dict[str, object]:
            try:
                receipt = run_progressive_bootstrap(
                    cast(ProgressiveIndexManager, manager),
                    unique_file_paths,
                    documents_roots=_bootstrap_documents_roots,
                )
            except Exception:
                logger.exception(
                    "Progressive bootstrap task failed for %d document(s)",
                    len(unique_file_paths),
                )
                return False
            logger.info(
                "Task completed: progressively indexed %d document(s)",
                receipt.successful,
            )
            if receipt.failed == 0:
                _persist_document_manifest(indexed_paths=unique_file_paths)
            records = getattr(getattr(receipt, "ingestion", None), "records", ())
            outcomes = [
                getattr(record, "status", None) == "committed"
                for record in records
            ]
            if len(outcomes) != len(unique_file_paths):
                return receipt.failed == 0
            return (
                True
                if all(outcomes)
                else _batch_result(outcomes, error="progressive batch item failed")
            )

        return _run_as_writer(
            _progressive_operation,
            operation_name="index_documents_batch",
            operation_args=(unique_file_paths,),
            busy_result=False,
        )

    def _operation() -> bool | dict[str, object]:
        completed_paths: list[str] = []
        failures: list[str] = []

        try:
            manager.index_documents(
                unique_file_paths,
                force=force,
                persist=True,
            )
            completed_paths = unique_file_paths
            _persist_document_manifest(indexed_paths=completed_paths)
        except Exception:
            logger.warning(
                "Batch index task failed; retrying files individually before one final persist",
                exc_info=True,
            )
            for file_path in unique_file_paths:
                try:
                    manager_result = manager.index_document(file_path, force=force)
                    logger.info(
                        "Batch index item manager result: path=%s force=%s result=%s failed_files=%s",
                        file_path,
                        force,
                        manager_result,
                        _failed_file_details(manager, file_path) if not manager_result else [],
                    )
                    completed_paths.append(file_path)
                except Exception:
                    failures.append(file_path)
                    logger.exception(
                        "Task failed within batch: index %s",
                        file_path,
                    )

            if completed_paths:
                try:
                    manager.persist()
                    _persist_document_manifest(indexed_paths=completed_paths)
                except Exception:
                    logger.exception(
                        "Batch fallback persist failed for %d indexed document(s)",
                        len(completed_paths),
                    )
                    return _batch_result(
                        [False] * len(unique_file_paths),
                        error="batch persist failed",
                    )

        if (
            completed_paths
            and _bootstrap_index_path is not None
            and _bootstrap_documents_roots
        ):
            mark_bootstrap_files_completed(
                _bootstrap_index_path,
                _bootstrap_documents_roots,
                completed_paths,
            )

        logger.info(
            "Task completed: indexed %d document(s) in batch%s",
            len(completed_paths),
            "" if not failures else f" with {len(failures)} failure(s)",
        )
        if not failures:
            return True
        return _batch_result(
            [file_path in completed_paths for file_path in unique_file_paths],
            error="batch item failed",
        )

    return _run_as_writer(
        _operation,
        operation_name="index_documents_batch",
        operation_args=(unique_file_paths,),
        busy_result=False,
    )

@_writer_owned_task(
    busy_result={
        "status": "error",
        "error": "index_writer_busy",
        "details": "Index writer is owned by a rebuild. Retry shortly.",
    }
)
def _index_records_batch(
    record_payloads: list[dict[str, object]],
) -> dict[str, object]:
    """Deserialize, index, and persist a Record batch in the worker.

    Record ingestion comes from separate applications, so keeping the
    entire write path in the worker prevents the daemon and worker from
    mutating the same persisted index concurrently.
    """
    if _index_manager is None:
        logger.error("IndexManager not available for record batch task")
        return {
            "status": "error",
            "error": "record_queue_unavailable",
            "details": "Index worker is not configured.",
        }
    from mcp_markdown_ragdocs.daemon.record_rpc import (
        RecordSerializationError,
        deserialize_record,
    )

    records = []
    for index, payload in enumerate(record_payloads):
        try:
            records.append(deserialize_record(payload))
        except RecordSerializationError as exc:
            return {
                "status": "error",
                "error": "invalid_record",
                "record_index": index,
                "details": str(exc),
            }

    if not records:
        return {"status": "ok", "indexed_count": 0}

    for index, record in enumerate(records):
        try:
            _index_manager.index_record(record)
        except Exception as exc:
            logger.exception(
                "Task failed within record batch at index %d",
                index,
            )
            return {
                "status": "error",
                "error": "record_indexing_failed",
                "record_index": index,
                "indexed_count": index,
                "details": str(exc),
            }

    try:
        _index_manager.persist()
    except Exception as exc:
        logger.exception("Task failed to persist record batch")
        return {
            "status": "error",
            "error": "record_indexing_failed",
            "indexed_count": len(records),
            "details": str(exc),
        }

    logger.info("Task completed: indexed %d record(s) in batch", len(records))
    return {"status": "ok", "indexed_count": len(records)}

@_writer_owned_task(busy_result=False)
def _remove_document(doc_id: str) -> bool:
    """Remove a document from all indices."""
    if _index_manager is None:
        logger.error("IndexManager not available for task execution")
        return False
    try:
        _index_manager.remove_document(doc_id)
        _index_manager.persist()
        _persist_document_manifest(removed_doc_ids=[doc_id])
        logger.info("Task completed: removed %s", doc_id)
        return True
    except Exception:
        logger.exception("Task failed: remove %s", doc_id)
        return False

@_writer_owned_task(busy_result=False)
def _remove_documents_batch(doc_ids: list[str]) -> bool | dict[str, object]:
    """Remove a burst of documents and persist once after the batch."""
    if _index_manager is None:
        logger.error("IndexManager not available for batch task execution")
        return False
    unique_doc_ids = list(dict.fromkeys(doc_ids))
    if not unique_doc_ids:
        return True

    removed_doc_ids: list[str] = []
    failures: list[str] = []

    try:
        _index_manager.remove_documents(unique_doc_ids, persist=True)
        removed_doc_ids = unique_doc_ids
        _persist_document_manifest(removed_doc_ids=removed_doc_ids)
    except Exception:
        logger.warning(
            "Batch remove task failed; retrying documents individually before one final persist",
            exc_info=True,
        )
        for doc_id in unique_doc_ids:
            try:
                _index_manager.remove_document(doc_id)
                removed_doc_ids.append(doc_id)
            except Exception:
                failures.append(doc_id)
                logger.exception(
                    "Task failed within batch: remove %s",
                    doc_id,
                )

        if removed_doc_ids:
            try:
                _index_manager.persist()
                _persist_document_manifest(removed_doc_ids=removed_doc_ids)
            except Exception:
                logger.exception(
                    "Batch fallback persist failed for %d removed document(s)",
                    len(removed_doc_ids),
                )
                return False

    logger.info(
        "Task completed: removed %d document(s) in batch%s",
        len(removed_doc_ids),
        "" if not failures else f" with {len(failures)} failure(s)",
    )
    if not failures:
        return True
    return _batch_result(
        [doc_id in removed_doc_ids for doc_id in unique_doc_ids],
        error="batch item failed",
    )

@_writer_owned_task(busy_result=False, on_busy=_defer_git_refresh)
def _refresh_git_repository(git_dir: str) -> bool:
    """Refresh the git index for one repository."""
    if _index_manager is None:
        logger.error("IndexManager not available for git refresh task")
        return False
    manager = _index_manager
    from mcp_markdown_ragdocs.adapters.sources.git import GitContentSource
    from mcp_markdown_ragdocs.config import (
        load_config,
        resolve_project_id_for_path,
    )
    from mcp_markdown_ragdocs.indexing.git_ingestion import (
        iter_git_ingestion_receipts,
    )
    git_dir_path = Path(git_dir).resolve()
    repo_key = str(git_dir_path)
    with _git_refresh_lock:
        if repo_key in _git_refresh_in_flight:
            logger.info("Skipping git refresh already running for %s", git_dir_path)
            return True
        _git_refresh_in_flight.add(repo_key)

    started_at = datetime.now(UTC)
    _save_git_refresh_progress(
        git_dir_path,
        state="running",
        started_at=started_at,
        processed_count=0,
        discovered_count=0,
        processed_chunk_count=0,
        discovered_chunk_count=0,
        error=None,
        completed_at=None,
    )
    processed_commit_ids: set[str] = set()
    discovered_commit_ids: set[str] = set()
    indexed = 0
    discovered = 0
    indexed_chunks = 0
    discovered_chunks = 0
    latest_cursor: int | None = None
    refresh_head: str | None = None
    newest_commit: str | None = None
    initial_embedding_metrics: dict[str, int] = {}
    try:
        refresh_head = get_git_ref_signature(git_dir_path)
        newest_commit = _newest_commit_from_ref_signature(refresh_head)
        _save_git_refresh_progress(
            git_dir_path,
            state="running",
            observed_head=refresh_head,
            newest_commit=newest_commit,
        )
        cursor = (
            get_cursor(_bootstrap_index_path, git_dir_path)
            if _bootstrap_index_path is not None
            else None
        )
        since = str(max(0, cursor - 1)) if cursor is not None else None
        config = getattr(manager, "_config", None) or load_config()
        source = GitContentSource(
            git_dir_path,
            workspace_id=resolve_project_id_for_path(git_dir_path.parent, config),
        )
        if (
            _bootstrap_index_path is not None
            and refresh_head is not None
            and cursor is not None
            and get_head(_bootstrap_index_path, git_dir_path) == refresh_head
        ):
            logger.debug("Skipping unchanged git repository %s", git_dir_path)
            _save_git_refresh_progress(
                git_dir_path,
                state="skipped",
                started_at=started_at,
                cursor=cursor,
                observed_head=refresh_head,
                newest_commit=newest_commit,
                processed_count=0,
                discovered_count=0,
                completed_at=datetime.now(UTC).isoformat(),
            )
            return True

        repair_attribution = getattr(
            manager,
            "reconcile_git_project_attribution",
            None,
        )
        repaired = (
            repair_attribution(git_dir_path, source.workspace_id)
            if repair_attribution is not None
            else 0
        )
        if repaired:
            manager.persist()
            logger.info(
                "Repaired project attribution for %d Git records in %s",
                repaired,
                git_dir_path.parent,
            )

        latest_cursor = cursor
        initial_embedding_metrics = _embedding_cache_metrics(manager)

        async def _ingest() -> None:
            nonlocal discovered, discovered_chunks, indexed, indexed_chunks, latest_cursor
            async for receipt in iter_git_ingestion_receipts(
                manager,
                source,
                since=since,
                batch_size=GIT_REFRESH_BATCH_SIZE,
            ):
                discovered_chunks += len(receipt.records)
                discovered_commit_ids.update(
                    commit_id
                    for result in receipt.records
                    if (commit_id := _git_commit_id_from_result(result)) is not None
                )
                result_success_flags = [
                    getattr(result, "successful", None) for result in receipt.records
                ]
                if any(isinstance(flag, bool) for flag in result_success_flags):
                    successful_results = [
                        result
                        for result, successful in zip(
                            receipt.records, result_success_flags, strict=True
                        )
                        if successful is True
                    ]
                else:
                    successful_results = (
                        list(receipt.records) if receipt.failed == 0 else []
                    )
                indexed_chunks += len(successful_results)
                processed_commit_ids.update(
                    commit_id
                    for result in successful_results
                    if (commit_id := _git_commit_id_from_result(result)) is not None
                )
                indexed = len(processed_commit_ids)
                discovered = len(discovered_commit_ids)
                if receipt.checkpoint is not None:
                    latest_cursor = max(latest_cursor or 0, int(receipt.checkpoint))
                current_embedding_metrics = _embedding_cache_metrics(manager)
                metric_deltas = {
                    key: value - initial_embedding_metrics.get(key, 0)
                    for key, value in current_embedding_metrics.items()
                }
                if receipt.failed:
                    _save_git_refresh_progress(
                        git_dir_path,
                        state="failed",
                        started_at=started_at,
                        cursor=latest_cursor,
                        observed_head=refresh_head,
                        newest_commit=newest_commit,
                        processed_count=indexed,
                        discovered_count=discovered,
                        processed_chunk_count=indexed_chunks,
                        discovered_chunk_count=discovered_chunks,
                        error=f"{receipt.failed} record(s) failed",
                        metrics=metric_deltas,
                    )
                    raise RuntimeError(
                        f"Git ingestion failed for {git_dir_path}: "
                        f"{receipt.failed} record(s)"
                    )
                manager.persist()
                if _bootstrap_index_path is not None and latest_cursor is not None:
                    save_cursor(_bootstrap_index_path, git_dir_path, latest_cursor)
                _save_git_refresh_progress(
                    git_dir_path,
                    state="running",
                    started_at=started_at,
                    cursor=latest_cursor,
                    observed_head=refresh_head,
                    newest_commit=newest_commit,
                    processed_count=indexed,
                    discovered_count=discovered,
                    processed_chunk_count=indexed_chunks,
                    discovered_chunk_count=discovered_chunks,
                    metrics=metric_deltas,
                )

        asyncio.run(_ingest())
        if _bootstrap_index_path is not None and refresh_head is not None:
            # Persist the head observed before ingestion. A commit created
            # during the task must remain visible to the next poll.
            save_head(_bootstrap_index_path, git_dir_path, refresh_head)
        final_embedding_metrics = _embedding_cache_metrics(manager)
        metric_deltas = {
            key: value - initial_embedding_metrics.get(key, 0)
            for key, value in final_embedding_metrics.items()
        }
        _save_git_refresh_progress(
            git_dir_path,
            state="completed",
            started_at=started_at,
            cursor=latest_cursor,
            observed_head=refresh_head,
            newest_commit=newest_commit,
            processed_count=indexed,
            discovered_count=discovered,
            processed_chunk_count=indexed_chunks,
            discovered_chunk_count=discovered_chunks,
            completed_at=datetime.now(UTC).isoformat(),
            error=None,
            metrics=metric_deltas,
        )
        logger.info(
            "Task completed: refreshed git repository %s (%d commits)",
            git_dir_path,
            indexed,
        )
        return True
    except Exception as error:
        final_embedding_metrics = _embedding_cache_metrics(manager)
        metric_deltas = {
            key: value - initial_embedding_metrics.get(key, 0)
            for key, value in final_embedding_metrics.items()
        }
        _save_git_refresh_progress(
            git_dir_path,
            state="failed",
            started_at=started_at,
            cursor=latest_cursor,
            observed_head=refresh_head,
            newest_commit=newest_commit,
            processed_count=indexed,
            discovered_count=discovered,
            processed_chunk_count=indexed_chunks,
            discovered_chunk_count=discovered_chunks,
            error=str(error),
            completed_at=datetime.now(UTC).isoformat(),
            metrics=metric_deltas,
        )
        logger.exception("Task failed: refresh git %s", git_dir_path)
        return False
    finally:
        with _git_refresh_lock:
            _git_refresh_in_flight.discard(repo_key)
            if repo_key not in _git_refresh_deferred:
                _git_refresh_pending.discard(repo_key)

def _rebuild_index(project_override: str | None, request_id: str) -> bool:
    """Run a daemon-owned rebuild inside the long-lived worker runtime."""
    if _index_manager is None:
        logger.error("IndexManager not available for rebuild task execution")
        return False
    if _bootstrap_index_path is None:
        logger.error("Runtime root not configured for rebuild task execution")
        return False
    runtime_root = _bootstrap_index_path
    manager = _index_manager

    from mcp_markdown_ragdocs.config import load_config

    def _operation() -> dict[str, object]:
        try:
            return run_rebuild(
                runtime_root=runtime_root,
                config=load_config(),
                index_manager=manager,
                global_documents_roots=_bootstrap_documents_roots,
                request_id=request_id,
                project_override=project_override,
                schedule_vocabulary_catch_up=None,
            )
        except Exception as exc:
            logger.exception("Task failed: rebuild index")
            return {"status": "failed", "error": str(exc)}

    payload = _run_as_writer(
        _operation,
        owner_token=request_id,
        busy_result={"status": "failed", "error": "index_writer_busy"},
    )
    if payload.get("error") != "index_writer_busy":
        payload = write_rebuild_status(
            runtime_root,
            {
                **payload,
                "writer_owned": False,
                "writer_owner": None,
            },
        )
    if payload.get("status") == "succeeded" and _schedule_vocabulary_catch_up is not None:
        try:
            scheduled = bool(_schedule_vocabulary_catch_up())
        except Exception:
            logger.warning(
                "Failed to schedule vocabulary catch-up after rebuild",
                exc_info=True,
            )
        else:
            if scheduled:
                write_rebuild_status(
                    runtime_root,
                    {
                        **payload,
                        "vocabulary_catch_up_scheduled": True,
                    },
                )
    return payload.get("status") == "succeeded"

@_writer_owned_task(
    busy_result={
        "status": "error",
        "error": "index_writer_busy",
        "details": "Index writer is owned by a rebuild. Retry shortly.",
    }
)
def _reindex_model(
    operation: str,
    model: str | None,
    truncate_dim: int | None,
    old_model: str | None,
    request_id: str,
) -> dict[str, object]:
    """Run one durable model-migration operation in the worker."""
    if _index_manager is None:
        return {"status": "error", "error": "reindex_queue_unavailable"}
    if _bootstrap_index_path is None:
        return {"status": "error", "error": "reindex_runtime_unavailable"}
    from mcp_markdown_ragdocs.config import load_config

    runtime_root = _bootstrap_index_path
    write_reindex_status(
        runtime_root,
        {
            "status": "running",
            "operation": operation,
            "request_id": request_id,
            "model": model,
            "truncate_dim": truncate_dim,
            "old_model": old_model,
            "phase": "running",
            "started_at": datetime.now(UTC).isoformat(),
            "error": None,
        },
    )
    try:
        config = getattr(_index_manager, "_config", None) or load_config()
        state = run_reindex_operation(
            config=config,
            index_path=runtime_root,
            runtime_root=runtime_root,
            operation=operation,
            model=model,
            truncate_dim=truncate_dim,
            old_model=old_model,
        )

        namespace = state.source if state.phase.value == "rollback" else state.target
        replace_vector_store = getattr(
            _index_manager,
            "replace_vector_store",
            None,
        )
        if callable(replace_vector_store):
            from mcp_markdown_ragdocs.context import ApplicationContext

            config.llm.embedding_model = namespace.model_name
            config.embedding.truncate_dim = namespace.dim
            replace_vector_store(
                ApplicationContext._build_vector_store(
                    config,
                    namespace.model_name,
                )
            )
        return {
            "status": "ok",
            "request_id": request_id,
            "phase": state.phase.value,
        }
    except Exception as exc:
        logger.exception("Task failed: model reindex")
        write_reindex_status(
            runtime_root,
            {
                "status": "failed",
                "operation": operation,
                "request_id": request_id,
                "model": model,
                "truncate_dim": truncate_dim,
                "old_model": old_model,
                "phase": "failed",
                "error": str(exc),
                "completed_at": datetime.now(UTC).isoformat(),
            },
        )
        return {
            "status": "error",
            "request_id": request_id,
            "error": str(exc),
        }

def register_tasks(
    huey: SqliteHuey,
    index_manager: IndexManagerLike,
    task_lease_store: TaskLeasePort,
    work_intent_store: WorkIntentPort,
    task_backpressure_limit: int = 100,
    bootstrap_index_path: Path | None = None,
    bootstrap_documents_roots: list[Path] | None = None,
    schedule_vocabulary_catch_up: Callable[[], bool] | None = None,
) -> None:
    """Register indexing tasks with the given Huey instance.

    Must be called before enqueuing tasks. Typically called during
    application startup when the worker is being configured.
    """
    global _huey, _index_manager, _task_backpressure_limit
    global _bootstrap_index_path, _bootstrap_documents_roots
    global _schedule_vocabulary_catch_up, _task_lease_store, _work_intent_store
    global index_document_task, index_documents_batch_task, index_records_batch_task
    global remove_document_task
    global remove_documents_batch_task, refresh_git_repository_task
    global rebuild_index_task, reindex_model_task
    global gdrive_inventory_task, gdrive_startup_task, gdrive_changes_task, gdrive_retry_task
    global gdrive_backfill_task, gdrive_lease_task, gdrive_watch_task
    global gdrive_health_task
    _huey = huey
    _index_manager = index_manager
    _task_backpressure_limit = max(1, task_backpressure_limit)
    _bootstrap_index_path = bootstrap_index_path
    _bootstrap_documents_roots = list(bootstrap_documents_roots or [])
    _schedule_vocabulary_catch_up = schedule_vocabulary_catch_up
    _task_lease_store = task_lease_store
    _work_intent_store = work_intent_store
    with _git_refresh_lock:
        _git_refresh_in_flight.clear()
        _git_refresh_pending.clear()
        _git_refresh_deferred.clear()

    registered_tasks = register_huey_tasks(
        huey,
        {
            "index_document": _intent_task("index_document")(_index_document),
            "index_documents_batch": _intent_task("index_documents_batch")(
                _index_documents_batch
            ),
            "index_records_batch": _intent_task("index_records_batch")(
                _index_records_batch
            ),
            "remove_document": _intent_task("remove_document")(_remove_document),
            "remove_documents_batch": _intent_task("remove_documents_batch")(
                _remove_documents_batch
            ),
            "refresh_git_repository": _intent_task("refresh_git_repository")(
                _refresh_git_repository
            ),
            "rebuild_index": _intent_task("rebuild_index")(_rebuild_index),
            "reindex_model": _intent_task("reindex_model")(_reindex_model),
        },
    )
    gdrive_runtime = (
        build_gdrive_task_runtime(
            index_manager,
            _task_lease_store,
            _work_intent_store,
        )
        if (
            isinstance(index_manager, GDriveTaskManager)
            and _task_lease_store is not None
            and _work_intent_store is not None
        )
        else None
    )
    gdrive_tasks = register_gdrive_tasks(huey, gdrive_runtime)
    index_document_task = registered_tasks["index_document"]
    index_documents_batch_task = registered_tasks["index_documents_batch"]
    index_records_batch_task = registered_tasks["index_records_batch"]
    remove_document_task = registered_tasks["remove_document"]
    remove_documents_batch_task = registered_tasks["remove_documents_batch"]
    refresh_git_repository_task = registered_tasks["refresh_git_repository"]
    rebuild_index_task = registered_tasks["rebuild_index"]
    reindex_model_task = registered_tasks["reindex_model"]
    gdrive_startup_task = gdrive_tasks.get("gdrive_startup")
    gdrive_inventory_task = gdrive_tasks.get("gdrive_inventory")
    gdrive_changes_task = gdrive_tasks.get("gdrive_changes")
    gdrive_retry_task = gdrive_tasks.get("gdrive_retry")
    gdrive_backfill_task = gdrive_tasks.get("gdrive_backfill")
    gdrive_lease_task = gdrive_tasks.get("gdrive_lease")
    gdrive_watch_task = gdrive_tasks.get("gdrive_watch")
    gdrive_health_task = gdrive_tasks.get("gdrive_health")
    logger.info("Indexing tasks registered with Huey")


def enqueue_index(file_path: str, force: bool = False) -> bool:
    """Enqueue an index_document task. Returns True if enqueued, False if no Huey."""
    return submit_index_request(file_path, force=force).enqueued


def submit_index_request(file_path: str, force: bool = False) -> TaskSubmissionResult:
    if index_document_task is None or _huey is None:
        return TaskSubmissionResult(status="unavailable")
    if _writer_is_active():
        return TaskSubmissionResult(status="already_pending")
    if is_backpressured(
        _huey,
        _task_backpressure_limit,
        item=file_path,
        warning_message="Skipping index enqueue for %s due to task queue backpressure (%d pending >= %d limit)",
    ):
        return TaskSubmissionResult(status="backpressured")
    claim = _intent_claim(
        "index_document",
        _canonical_document_identity(file_path),
        {"file_path": file_path, "force": force},
        force_reopen=force,
    )
    if claim is None:
        return TaskSubmissionResult(status="already_pending")
    intent, claim_token = claim
    try:
        index_document_task(
            file_path,
            force=force,
            intent_id=intent.intent_id,
            claim_token=claim_token,
        )
    except Exception:
        if _intent_store() is not None:
            _release_intent(intent.intent_id, claim_token)
        raise
    return TaskSubmissionResult(status="enqueued")


def _get_pending_task_first_args(task_name: str) -> set[str]:
    """Return first positional string args already pending for the given task name."""
    return get_pending_task_first_args(
        _huey,
        task_name,
        inspection_failure_log_message="Failed to inspect pending Huey tasks; startup batch dedupe disabled",
        deserialize_failure_log_message="Failed to deserialize pending Huey task while inspecting startup queue",
    )


def _get_pending_index_document_paths() -> set[str]:
    """Return file paths already pending in single or batch index tasks."""

    def _extract_values(task: object) -> set[str]:
        args = getattr(task, "args", ())
        if not args:
            return set()

        first_arg = args[0]
        if isinstance(first_arg, str):
            return {first_arg}
        if isinstance(first_arg, list):
            return {item for item in first_arg if isinstance(item, str)}
        return set()

    return get_pending_task_values(
        _huey,
        {"_index_document", "_index_documents_batch"},
        value_extractor=lambda task: {
            _canonical_document_identity(value)
            for value in _extract_values(task)
        },
        inspection_failure_log_message="Failed to inspect pending Huey tasks; startup batch dedupe disabled",
        deserialize_failure_log_message="Failed to deserialize pending Huey task while inspecting startup queue",
    )


def _get_pending_refresh_git_dirs() -> set[str]:
    """Return git dirs already pending in the queue for _refresh_git_repository tasks."""
    return _get_pending_task_first_args("_refresh_git_repository")


def _get_pending_remove_doc_ids() -> set[str]:
    """Return doc IDs already pending in single or batch remove tasks."""

    def _extract_values(task: object) -> set[str]:
        args = getattr(task, "args", ())
        if not args:
            return set()

        first_arg = args[0]
        if isinstance(first_arg, str):
            return {first_arg}
        if isinstance(first_arg, list):
            return {item for item in first_arg if isinstance(item, str)}
        return set()

    return get_pending_task_values(
        _huey,
        {"_remove_document", "_remove_documents_batch"},
        value_extractor=lambda task: {
            _canonical_document_identity(value)
            for value in _extract_values(task)
        },
        inspection_failure_log_message="Failed to inspect pending Huey tasks; steady-state remove dedupe disabled",
        deserialize_failure_log_message="Failed to deserialize pending Huey task while inspecting remove queue",
    )


def _submit_backpressure_limited_batch_request(
    *,
    task_submitter: Callable[..., object],
    batch_name: str,
    items: list[str],
    task_kwargs: dict[str, object] | None = None,
    pending_items: set[str] | None = None,
) -> TaskBatchSubmissionResult:
    if _huey is None:
        return TaskBatchSubmissionResult(
            queue_available=False,
            requested_unique_count=len(set(items)),
            enqueued_count=0,
        )

    unique_items = list(dict.fromkeys(items))
    requested_unique_count = len(unique_items)
    remaining_items, already_pending_count = coalesce_pending_first_args(
        unique_items,
        pending_first_args=pending_items,
    )

    if not remaining_items:
        return TaskBatchSubmissionResult(
            queue_available=True,
            requested_unique_count=requested_unique_count,
            enqueued_count=0,
            already_pending_count=already_pending_count,
        )

    if is_backpressured(
        _huey,
        _task_backpressure_limit,
        item=f"{len(remaining_items)} {batch_name}(s)",
        warning_message="Skipping %s batch enqueue due to task queue backpressure (%d pending >= %d limit)",
    ):
        return TaskBatchSubmissionResult(
            queue_available=True,
            requested_unique_count=requested_unique_count,
            enqueued_count=0,
            already_pending_count=already_pending_count,
            backpressured_items=tuple(remaining_items),
        )

    enqueued_count, skipped_pending_count = submit_coalesced_batch_task(
        task_submitter,
        remaining_items,
        task_kwargs=task_kwargs,
    )
    return TaskBatchSubmissionResult(
        queue_available=True,
        requested_unique_count=requested_unique_count,
        enqueued_count=enqueued_count,
        already_pending_count=already_pending_count + skipped_pending_count,
    )


def enqueue_index_batch(file_paths: list[str], force: bool = False) -> int:
    """Enqueue many index tasks without watcher backpressure throttling.

    Intended for cold-start/bootstrap flows where the full corpus needs to be
    materialized durably by the worker.
    """
    return submit_index_batch(file_paths, force=force).enqueued_count


def submit_index_batch(
    file_paths: list[str],
    force: bool = False,
    progressive: bool = False,
) -> TaskBatchSubmissionResult:
    if index_documents_batch_task is None or _huey is None:
        return TaskBatchSubmissionResult(
            queue_available=False,
            requested_unique_count=len(_unique_document_paths(file_paths)),
            enqueued_count=0,
        )
    unique_file_paths = _unique_document_paths(file_paths)
    if _writer_is_active():
        unique_count = len(unique_file_paths)
        return TaskBatchSubmissionResult(
            queue_available=True,
            requested_unique_count=unique_count,
            enqueued_count=0,
            already_pending_count=unique_count,
        )

    pending_paths = set() if force else _get_pending_index_document_paths()
    requested_unique_paths = {
        _canonical_document_identity(file_path) for file_path in unique_file_paths
    }
    remaining_paths = [
        file_path
        for file_path in unique_file_paths
        if force or _canonical_document_identity(file_path) not in pending_paths
    ]
    already_pending_count = sum(
        1 for file_path in requested_unique_paths if file_path in pending_paths
    )

    enqueued_count = 0
    keyed_claims, skipped_count = _intent_claim_batch(
        "index_document",
        [
            (
                _canonical_document_identity(file_path),
                {"file_path": file_path, "force": force},
            )
            for file_path in remaining_paths
        ],
        force_reopen=force,
    )
    already_pending_count += skipped_count
    claim_by_key = {key: claim for key, claim in keyed_claims}
    claims = list(claim_by_key.values())
    claim_paths = [
        file_path
        for file_path in remaining_paths
        if _canonical_document_identity(file_path) in claim_by_key
    ]
    if claims:
        submitted_claims: list[tuple[str, str]] = []
        try:
            for start in range(0, len(claim_paths), DOCUMENT_TASK_BATCH_SIZE):
                batch_paths = claim_paths[start : start + DOCUMENT_TASK_BATCH_SIZE]
                batch_claims = [
                    claim_by_key[_canonical_document_identity(file_path)]
                    for file_path in batch_paths
                ]
                kwargs: dict[str, object] = {
                    "force": force,
                    "intent_claims": batch_claims,
                }
                if progressive:
                    kwargs["progressive"] = True
                index_documents_batch_task(batch_paths, **kwargs)
                submitted_claims.extend(batch_claims)
        except Exception:
            if _intent_store() is not None:
                for intent_id, claim_token in claims:
                    if (intent_id, claim_token) not in submitted_claims:
                        _release_intent(intent_id, claim_token)
            raise
        enqueued_count = len(claims)

    if already_pending_count > 0:
        logger.info(
            "Skipped %d startup indexing task(s) already pending in queue",
            already_pending_count,
        )

    return TaskBatchSubmissionResult(
        queue_available=True,
        requested_unique_count=len(requested_unique_paths),
        enqueued_count=enqueued_count,
        already_pending_count=already_pending_count,
    )


def submit_index_request_batch(
    file_paths: list[str],
    force: bool = False,
) -> TaskBatchSubmissionResult:
    if index_documents_batch_task is None or _huey is None:
        return TaskBatchSubmissionResult(
            queue_available=False,
            requested_unique_count=len(_unique_document_paths(file_paths)),
            enqueued_count=0,
        )
    unique_paths = _unique_document_paths(file_paths)
    pending_paths = set() if force else _get_pending_index_document_paths()
    remaining_paths = [
        file_path
        for file_path in unique_paths
        if force or file_path not in pending_paths
    ]
    already_pending_count = len(unique_paths) - len(remaining_paths)
    if is_backpressured(
        _huey,
        _task_backpressure_limit,
        item=f"{len(remaining_paths)} file(s)",
        warning_message=(
            "Skipping %s due to task queue backpressure "
            "(%d pending >= %d limit)"
        ),
    ):
        return TaskBatchSubmissionResult(
            queue_available=True,
            requested_unique_count=len(unique_paths),
            enqueued_count=0,
            already_pending_count=already_pending_count,
            backpressured_items=tuple(remaining_paths),
        )
    submission = submit_index_batch(remaining_paths, force=force)
    return TaskBatchSubmissionResult(
        queue_available=True,
        requested_unique_count=len(unique_paths),
        enqueued_count=submission.enqueued_count,
        already_pending_count=already_pending_count
        + submission.already_pending_count,
        backpressured_items=submission.backpressured_items,
    )


def submit_record_batch(
    record_payloads: list[dict[str, object]],
) -> object | None:
    """Queue a Record batch and return its Huey result handle."""
    if index_records_batch_task is None or _huey is None:
        return None
    if _writer_is_active():
        return _RejectedRecordBatch()
    canonical_key = hashlib.sha256(
        json.dumps(record_payloads, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    claim = _intent_claim(
        "index_records_batch",
        canonical_key,
        {"record_count": len(record_payloads)},
    )
    if claim is None:
        return _RejectedRecordBatch()
    intent, claim_token = claim
    try:
        return index_records_batch_task(
            record_payloads,
            priority=RECORD_BATCH_TASK_PRIORITY,
            intent_id=intent.intent_id,
            claim_token=claim_token,
        )
    except Exception:
        if _intent_store() is not None:
            _release_intent(intent.intent_id, claim_token)
        raise


def get_pending_index_document_count(file_paths: list[str]) -> int:
    """Count how many of the given file paths are already pending in Huey."""
    if not file_paths:
        return 0

    pending_paths = _get_pending_index_document_paths()
    unique_paths = {
        _canonical_document_identity(file_path) for file_path in file_paths
    }
    return sum(1 for file_path in unique_paths if file_path in pending_paths)


def enqueue_remove(doc_id: str) -> bool:
    """Enqueue a remove_document task. Returns True if enqueued, False if no Huey."""
    return submit_remove_request(doc_id).enqueued


def submit_remove_request(doc_id: str) -> TaskSubmissionResult:
    if remove_document_task is None or _huey is None:
        return TaskSubmissionResult(status="unavailable")
    if _writer_is_active():
        return TaskSubmissionResult(status="already_pending")
    if is_backpressured(
        _huey,
        _task_backpressure_limit,
        item=doc_id,
        warning_message="Skipping remove enqueue for %s due to task queue backpressure (%d pending >= %d limit)",
    ):
        return TaskSubmissionResult(status="backpressured")
    claim = _intent_claim(
        "remove_document",
        _canonical_document_identity(doc_id),
        {"doc_id": doc_id},
    )
    if claim is None:
        return TaskSubmissionResult(status="already_pending")
    intent, claim_token = claim
    try:
        remove_document_task(
            doc_id,
            intent_id=intent.intent_id,
            claim_token=claim_token,
        )
    except Exception:
        if _intent_store() is not None:
            _release_intent(intent.intent_id, claim_token)
        raise
    return TaskSubmissionResult(status="enqueued")


def submit_remove_request_batch(doc_ids: list[str]) -> TaskBatchSubmissionResult:
    if remove_documents_batch_task is None:
        return TaskBatchSubmissionResult(
            queue_available=False,
            requested_unique_count=len(set(doc_ids)),
            enqueued_count=0,
        )
    if _writer_is_active():
        unique_count = len(set(doc_ids))
        return TaskBatchSubmissionResult(
            queue_available=True,
            requested_unique_count=unique_count,
            enqueued_count=0,
            already_pending_count=unique_count,
        )

    unique_doc_ids = list(dict.fromkeys(doc_ids))
    pending_doc_ids = _get_pending_remove_doc_ids()
    remaining_doc_ids = [
        doc_id
        for doc_id in unique_doc_ids
        if _canonical_document_identity(doc_id) not in pending_doc_ids
    ]
    already_pending_count = len(unique_doc_ids) - len(remaining_doc_ids)
    if not remaining_doc_ids:
        return TaskBatchSubmissionResult(
            queue_available=True,
            requested_unique_count=len(unique_doc_ids),
            enqueued_count=0,
            already_pending_count=already_pending_count,
        )
    if is_backpressured(
        _huey,
        _task_backpressure_limit,
        item=f"{len(remaining_doc_ids)} document(s)",
        warning_message="Skipping %s due to task queue backpressure (%d pending >= %d limit)",
    ):
        return TaskBatchSubmissionResult(
            queue_available=True,
            requested_unique_count=len(unique_doc_ids),
            enqueued_count=0,
            already_pending_count=already_pending_count,
            backpressured_items=tuple(remaining_doc_ids),
        )
    keyed_claims, skipped_count = _intent_claim_batch(
        "remove_document",
        [
            (
                _canonical_document_identity(doc_id),
                {"doc_id": doc_id},
            )
            for doc_id in remaining_doc_ids
        ],
    )
    already_pending_count += skipped_count
    claims = [claim for _, claim in keyed_claims]
    claim_keys = {key for key, _ in keyed_claims}
    task_doc_ids = [
        doc_id
        for doc_id in remaining_doc_ids
        if _canonical_document_identity(doc_id) in claim_keys
    ]
    if not claims:
        return TaskBatchSubmissionResult(
            queue_available=True,
            requested_unique_count=len(unique_doc_ids),
            enqueued_count=0,
            already_pending_count=already_pending_count,
        )
    claim_by_key = {key: claim for key, claim in keyed_claims}
    submitted_claims: list[tuple[str, str]] = []
    try:
        for start in range(0, len(task_doc_ids), DOCUMENT_TASK_BATCH_SIZE):
            batch_doc_ids = task_doc_ids[start : start + DOCUMENT_TASK_BATCH_SIZE]
            batch_claims = [
                claim_by_key[_canonical_document_identity(doc_id)]
                for doc_id in batch_doc_ids
            ]
            remove_documents_batch_task(batch_doc_ids, intent_claims=batch_claims)
            submitted_claims.extend(batch_claims)
    except Exception:
        if _intent_store() is not None:
            for intent_id, claim_token in claims:
                if (intent_id, claim_token) not in submitted_claims:
                    _release_intent(intent_id, claim_token)
        raise
    return TaskBatchSubmissionResult(
        queue_available=True,
        requested_unique_count=len(unique_doc_ids),
        enqueued_count=len(task_doc_ids),
        already_pending_count=already_pending_count,
    )


def enqueue_refresh_git(git_dir: str) -> bool:
    """Enqueue a refresh_git_repository task. Returns True if enqueued."""
    return submit_refresh_git_request(git_dir).enqueued


def submit_refresh_git_request(git_dir: str) -> TaskSubmissionResult:
    if refresh_git_repository_task is None or _huey is None:
        return TaskSubmissionResult(status="unavailable")
    git_dir_key = _git_refresh_key(git_dir)
    with _git_refresh_lock:
        if (
            git_dir_key in _git_refresh_in_flight
            or git_dir_key in _git_refresh_pending
            or git_dir_key in _git_refresh_deferred
        ):
            logger.info(
                "Skipping git refresh enqueue for %s because it is already queued or running",
                git_dir,
            )
            return TaskSubmissionResult(status="already_pending")
        if _writer_is_active():
            _git_refresh_pending.add(git_dir_key)
            _git_refresh_deferred.add(git_dir_key)
            _save_git_refresh_progress(
                Path(git_dir),
                state="queued",
                queued_at=datetime.now(UTC).isoformat(),
            )
            logger.info("Deferring git refresh while rebuild owns the writer: %s", git_dir)
            return TaskSubmissionResult(status="already_pending")
        _git_refresh_pending.add(git_dir_key)

    if is_backpressured(
        _huey,
        _task_backpressure_limit,
        item=git_dir,
        warning_message="Skipping git refresh enqueue for %s due to task queue backpressure (%d pending >= %d limit)",
    ):
        with _git_refresh_lock:
            _git_refresh_pending.discard(git_dir_key)
        _save_git_refresh_progress(
            Path(git_dir),
            state="backpressured",
            error="task queue backpressure",
            completed_at=datetime.now(UTC).isoformat(),
        )
        return TaskSubmissionResult(status="backpressured")
    claim = _intent_claim(
        "refresh_git_repository",
        _git_refresh_key(git_dir),
        {"git_dir": git_dir},
    )
    if claim is None:
        with _git_refresh_lock:
            _git_refresh_pending.discard(git_dir_key)
        return TaskSubmissionResult(status="already_pending")
    intent, claim_token = claim
    try:
        refresh_git_repository_task(
            git_dir,
            priority=GIT_REFRESH_TASK_PRIORITY,
            intent_id=intent.intent_id,
            claim_token=claim_token,
        )
    except Exception:
        if _intent_store() is not None:
            _release_intent(intent.intent_id, claim_token)
        with _git_refresh_lock:
            _git_refresh_pending.discard(git_dir_key)
        raise
    if claim:
        _save_git_refresh_progress(
            Path(git_dir),
            state="queued",
            queued_at=datetime.now(UTC).isoformat(),
        )
        return TaskSubmissionResult(status="enqueued")
    return TaskSubmissionResult(status="already_pending")


def enqueue_refresh_git_batch(git_dirs: list[str]) -> int:
    """Enqueue many git refresh tasks, respecting queue backpressure."""
    return submit_refresh_git_batch(git_dirs).enqueued_count


def submit_refresh_git_batch(git_dirs: list[str]) -> TaskBatchSubmissionResult:
    if refresh_git_repository_task is None or _huey is None:
        return TaskBatchSubmissionResult(
            queue_available=False,
            requested_unique_count=len(set(git_dirs)),
            enqueued_count=0,
        )

    unique_git_dirs = list(dict.fromkeys(git_dirs))
    enqueued_count = 0
    already_pending_count = 0
    backpressured_items: list[str] = []
    for git_dir in unique_git_dirs:
        submission = submit_refresh_git_request(git_dir)
        if submission.status == "enqueued":
            enqueued_count += 1
        elif submission.status == "already_pending":
            already_pending_count += 1
        elif submission.status == "backpressured":
            backpressured_items.append(git_dir)

    return TaskBatchSubmissionResult(
        queue_available=True,
        requested_unique_count=len(unique_git_dirs),
        enqueued_count=enqueued_count,
        already_pending_count=already_pending_count,
        backpressured_items=tuple(backpressured_items),
    )


def submit_rebuild_request(
    project_override: str | None,
    *,
    request_id: str,
) -> TaskSubmissionResult:
    if rebuild_index_task is None or _huey is None:
        return TaskSubmissionResult(status="unavailable")

    writer_store = _writer_lease_store()
    if writer_store is None:
        return TaskSubmissionResult(status="unavailable")
    if not writer_store.acquire_writer(request_id):
        return TaskSubmissionResult(status="backpressured")

    queue_item = project_override or "__global__"
    if is_backpressured(
        _huey,
        _task_backpressure_limit,
        item=queue_item,
        warning_message="Skipping rebuild enqueue for %s due to task queue backpressure (%d pending >= %d limit)",
    ):
        writer_store.release_writer(request_id)
        return TaskSubmissionResult(status="backpressured")

    claim = _intent_claim(
        "rebuild_index",
        f"{project_override or '__global__'}:{request_id}",
        {"project_override": project_override, "request_id": request_id},
    )
    if claim is None:
        writer_store.release_writer(request_id)
        return TaskSubmissionResult(status="already_pending")
    intent, claim_token = claim
    try:
        rebuild_index_task(
            project_override,
            request_id=request_id,
            intent_id=intent.intent_id,
            claim_token=claim_token,
        )
    except Exception:
        if _intent_store() is not None:
            _release_intent(intent.intent_id, claim_token)
        writer_store.release_writer(request_id)
        raise
    return TaskSubmissionResult(status="enqueued")


def submit_reindex_request(
    operation: str,
    *,
    model: str | None,
    truncate_dim: int | None,
    old_model: str | None,
    request_id: str,
) -> TaskSubmissionResult:
    if reindex_model_task is None or _huey is None:
        return TaskSubmissionResult(status="unavailable")
    if _writer_is_active():
        return TaskSubmissionResult(status="already_pending")
    if is_backpressured(
        _huey,
        _task_backpressure_limit,
        item=f"reindex:{operation}",
        warning_message="Skipping reindex enqueue for %s due to task queue backpressure (%d pending >= %d limit)",
    ):
        return TaskSubmissionResult(status="backpressured")
    claim = _intent_claim(
        "reindex_model",
        f"{operation}:{model}:{truncate_dim}:{old_model}",
        {
            "operation": operation,
            "model": model,
            "truncate_dim": truncate_dim,
            "old_model": old_model,
            "request_id": request_id,
        },
    )
    if claim is None:
        return TaskSubmissionResult(status="already_pending")
    intent, claim_token = claim
    try:
        reindex_model_task(
            operation,
            model,
            truncate_dim,
            old_model,
            request_id=request_id,
            priority=REINDEX_TASK_PRIORITY,
            intent_id=intent.intent_id,
            claim_token=claim_token,
        )
    except Exception:
        if _intent_store() is not None:
            _release_intent(intent.intent_id, claim_token)
        raise
    return TaskSubmissionResult(status="enqueued")


def get_pending_task_count() -> int:
    return get_shared_pending_task_count(_huey)


def is_task_queue_available() -> bool:
    return _huey is not None and index_document_task is not None
