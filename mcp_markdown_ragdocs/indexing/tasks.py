"""Huey task definitions for indexing operations."""

from __future__ import annotations

import asyncio
import logging
import threading
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from mcp_markdown_ragdocs.coordination.task_submission import (
    coalesce_pending_first_args,
    get_pending_task_first_args,
    get_pending_task_values,
    is_backpressured,
    submit_coalesced_batch_task,
    submit_single_task,
)
from mcp_markdown_ragdocs.coordination.task_submission import (
    get_pending_task_count as get_shared_pending_task_count,
)
from mcp_markdown_ragdocs.git.repository import get_git_ref_signature
from searchkernel.api import (
    AsyncIndexIngestor,
    mark_bootstrap_file_completed,
    mark_bootstrap_files_completed,
    TaskBatchSubmissionResult,
    TaskSubmissionResult,
)
from mcp_markdown_ragdocs.indexing.git_refresh_state import (
    get_cursor,
    get_head,
    save_cursor,
    save_head,
)
from mcp_markdown_ragdocs.indexing.rebuild_service import run_rebuild
from mcp_markdown_ragdocs.indexing.reindex import (
    run_reindex_operation,
    write_reindex_status,
)

if TYPE_CHECKING:
    from huey import SqliteHuey

    from searchkernel.api import Record

logger = logging.getLogger(__name__)

RECORD_BATCH_TASK_PRIORITY = 100
GIT_REFRESH_TASK_PRIORITY = 10
REINDEX_TASK_PRIORITY = 200

__all__ = [
    "TaskBatchSubmissionResult",
    "TaskSubmissionResult",
]


class IndexManagerLike(Protocol):
    """Structural type for objects that can index/remove documents."""

    def index_document(self, file_path: str, force: bool = False) -> None: ...
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
    def index_record(self, record: Record) -> bool: ...


# Module-level references set during initialization
_huey: SqliteHuey | None = None
_index_manager: IndexManagerLike | None = None
_task_backpressure_limit: int = 100
_bootstrap_index_path: Path | None = None
_bootstrap_documents_roots: list[Path] = []
_schedule_vocabulary_catch_up: Callable[[], bool] | None = None
_git_refresh_in_flight: set[str] = set()
_git_refresh_lock = threading.Lock()

# Task references (set after register_tasks is called)
index_document_task = None
index_documents_batch_task = None
index_records_batch_task = None
remove_document_task = None
remove_documents_batch_task = None
refresh_git_repository_task = None
rebuild_index_task = None
reindex_model_task = None


def register_tasks(
    huey: SqliteHuey,
    index_manager: IndexManagerLike,
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
    global _schedule_vocabulary_catch_up
    global index_document_task, index_documents_batch_task, index_records_batch_task
    global remove_document_task
    global remove_documents_batch_task, refresh_git_repository_task
    global rebuild_index_task, reindex_model_task
    _huey = huey
    _index_manager = index_manager
    _task_backpressure_limit = max(1, task_backpressure_limit)
    _bootstrap_index_path = bootstrap_index_path
    _bootstrap_documents_roots = list(bootstrap_documents_roots or [])
    _schedule_vocabulary_catch_up = schedule_vocabulary_catch_up

    @huey.task()
    def _index_document(file_path: str, force: bool = False) -> bool:
        """Index or re-index a single document."""
        if _index_manager is None:
            logger.error("IndexManager not available for task execution")
            return False
        try:
            _index_manager.index_document(file_path, force=force)
            _index_manager.persist()
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

    @huey.task()
    def _index_documents_batch(file_paths: list[str], force: bool = False) -> bool:
        """Index a burst of documents and persist once after the batch."""
        if _index_manager is None:
            logger.error("IndexManager not available for batch task execution")
            return False

        unique_file_paths = list(dict.fromkeys(file_paths))
        if not unique_file_paths:
            return True

        completed_paths: list[str] = []
        failures: list[str] = []

        try:
            _index_manager.index_documents(
                unique_file_paths,
                force=force,
                persist=True,
            )
            completed_paths = unique_file_paths
        except Exception:
            logger.warning(
                "Batch index task failed; retrying files individually before one final persist",
                exc_info=True,
            )
            for file_path in unique_file_paths:
                try:
                    _index_manager.index_document(file_path, force=force)
                    completed_paths.append(file_path)
                except Exception:
                    failures.append(file_path)
                    logger.exception(
                        "Task failed within batch: index %s",
                        file_path,
                    )

            if completed_paths:
                try:
                    _index_manager.persist()
                except Exception:
                    logger.exception(
                        "Batch fallback persist failed for %d indexed document(s)",
                        len(completed_paths),
                    )
                    return False

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
        return not failures

    @huey.task()
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

    @huey.task()
    def _remove_document(doc_id: str) -> bool:
        """Remove a document from all indices."""
        if _index_manager is None:
            logger.error("IndexManager not available for task execution")
            return False
        try:
            _index_manager.remove_document(doc_id)
            _index_manager.persist()
            logger.info("Task completed: removed %s", doc_id)
            return True
        except Exception:
            logger.exception("Task failed: remove %s", doc_id)
            return False

    @huey.task()
    def _remove_documents_batch(doc_ids: list[str]) -> bool:
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
        return not failures

    @huey.task()
    def _refresh_git_repository(git_dir: str) -> bool:
        """Refresh the git index for one repository."""
        if _index_manager is None:
            logger.error("IndexManager not available for git refresh task")
            return False

        from mcp_markdown_ragdocs.adapters.sources.git import GitContentSource
        git_dir_path = Path(git_dir).resolve()
        repo_key = str(git_dir_path)
        with _git_refresh_lock:
            if repo_key in _git_refresh_in_flight:
                logger.info("Skipping git refresh already running for %s", git_dir_path)
                return True
            _git_refresh_in_flight.add(repo_key)

        try:
            refresh_head = get_git_ref_signature(git_dir_path)
            if (
                _bootstrap_index_path is not None
                and refresh_head is not None
                and get_head(_bootstrap_index_path, git_dir_path) == refresh_head
            ):
                logger.debug("Skipping unchanged git repository %s", git_dir_path)
                return True

            cursor = (
                get_cursor(_bootstrap_index_path, git_dir_path)
                if _bootstrap_index_path is not None
                else None
            )
            since = str(max(0, cursor - 1)) if cursor is not None else None
            source = GitContentSource(git_dir_path)
            records = source.iter_records(since=since)
            records = list(records)
            latest_cursor = max(
                (int(record.updated_at.timestamp()) for record in records),
                default=cursor,
            )
            receipt = asyncio.run(
                AsyncIndexIngestor(_index_manager).index_records(
                    records,
                    checkpoint=since,
                )
            )
            indexed = len(receipt.records)
            if indexed:
                _index_manager.persist()
                if _bootstrap_index_path is not None and latest_cursor is not None:
                    save_cursor(_bootstrap_index_path, git_dir_path, latest_cursor)
            if _bootstrap_index_path is not None and refresh_head is not None:
                # Persist the head observed before ingestion. A commit created
                # during the task must remain visible to the next poll.
                save_head(_bootstrap_index_path, git_dir_path, refresh_head)
            logger.info(
                "Task completed: refreshed git repository %s (%d commits)",
                git_dir_path,
                indexed,
            )
            return True
        except Exception:
            logger.exception("Task failed: refresh git %s", git_dir_path)
            return False
        finally:
            with _git_refresh_lock:
                _git_refresh_in_flight.discard(repo_key)

    @huey.task()
    def _rebuild_index(project_override: str | None, request_id: str) -> bool:
        """Run a daemon-owned rebuild inside the long-lived worker runtime."""
        if _index_manager is None:
            logger.error("IndexManager not available for rebuild task execution")
            return False
        if _bootstrap_index_path is None:
            logger.error("Runtime root not configured for rebuild task execution")
            return False

        from mcp_markdown_ragdocs.config import load_config

        try:
            payload = run_rebuild(
                runtime_root=_bootstrap_index_path,
                config=load_config(),
                index_manager=_index_manager,
                global_documents_roots=_bootstrap_documents_roots,
                request_id=request_id,
                project_override=project_override,
                schedule_vocabulary_catch_up=_schedule_vocabulary_catch_up,
            )
            return payload.get("status") == "succeeded"
        except Exception:
            logger.exception("Task failed: rebuild index")
            return False

    @huey.task()
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

    index_document_task = _index_document
    index_documents_batch_task = _index_documents_batch
    index_records_batch_task = _index_records_batch
    remove_document_task = _remove_document
    remove_documents_batch_task = _remove_documents_batch
    refresh_git_repository_task = _refresh_git_repository
    rebuild_index_task = _rebuild_index
    reindex_model_task = _reindex_model
    logger.info("Indexing tasks registered with Huey")


def enqueue_index(file_path: str, force: bool = False) -> bool:
    """Enqueue an index_document task. Returns True if enqueued, False if no Huey."""
    return submit_index_request(file_path, force=force).enqueued


def submit_index_request(file_path: str, force: bool = False) -> TaskSubmissionResult:
    if index_document_task is None or _huey is None:
        return TaskSubmissionResult(status="unavailable")
    if is_backpressured(
        _huey,
        _task_backpressure_limit,
        item=file_path,
        warning_message="Skipping index enqueue for %s due to task queue backpressure (%d pending >= %d limit)",
    ):
        return TaskSubmissionResult(status="backpressured")
    enqueued = submit_single_task(
        index_document_task,
        file_path,
        task_kwargs={"force": force},
    )
    if enqueued:
        return TaskSubmissionResult(status="enqueued")
    return TaskSubmissionResult(status="already_pending")


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
        value_extractor=_extract_values,
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
        value_extractor=_extract_values,
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
) -> TaskBatchSubmissionResult:
    if index_documents_batch_task is None or _huey is None:
        return TaskBatchSubmissionResult(
            queue_available=False,
            requested_unique_count=len(set(file_paths)),
            enqueued_count=0,
        )

    unique_file_paths = list(dict.fromkeys(file_paths))
    pending_paths = set() if force else _get_pending_index_document_paths()
    requested_unique_paths = set(unique_file_paths)
    remaining_paths = [
        file_path
        for file_path in unique_file_paths
        if force or file_path not in pending_paths
    ]
    already_pending_count = sum(
        1 for file_path in requested_unique_paths if file_path in pending_paths
    )

    enqueued_count = 0
    if remaining_paths:
        index_documents_batch_task(remaining_paths, force=force)
        enqueued_count = len(remaining_paths)

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
    if index_documents_batch_task is None:
        return TaskBatchSubmissionResult(
            queue_available=False,
            requested_unique_count=len(set(file_paths)),
            enqueued_count=0,
        )

    pending_paths = set() if force else _get_pending_index_document_paths()
    return _submit_backpressure_limited_batch_request(
        task_submitter=index_documents_batch_task,
        batch_name="file",
        items=file_paths,
        task_kwargs={"force": force},
        pending_items=pending_paths,
    )


def submit_record_batch(
    record_payloads: list[dict[str, object]],
) -> object | None:
    """Queue a Record batch and return its Huey result handle."""
    if index_records_batch_task is None or _huey is None:
        return None
    return index_records_batch_task(
        record_payloads,
        priority=RECORD_BATCH_TASK_PRIORITY,
    )


def get_pending_index_document_count(file_paths: list[str]) -> int:
    """Count how many of the given file paths are already pending in Huey."""
    if not file_paths:
        return 0

    pending_paths = _get_pending_index_document_paths()
    unique_paths = set(file_paths)
    return sum(1 for file_path in unique_paths if file_path in pending_paths)


def enqueue_remove(doc_id: str) -> bool:
    """Enqueue a remove_document task. Returns True if enqueued, False if no Huey."""
    return submit_remove_request(doc_id).enqueued


def submit_remove_request(doc_id: str) -> TaskSubmissionResult:
    if remove_document_task is None or _huey is None:
        return TaskSubmissionResult(status="unavailable")
    if is_backpressured(
        _huey,
        _task_backpressure_limit,
        item=doc_id,
        warning_message="Skipping remove enqueue for %s due to task queue backpressure (%d pending >= %d limit)",
    ):
        return TaskSubmissionResult(status="backpressured")
    enqueued = submit_single_task(remove_document_task, doc_id)
    if enqueued:
        return TaskSubmissionResult(status="enqueued")
    return TaskSubmissionResult(status="already_pending")


def submit_remove_request_batch(doc_ids: list[str]) -> TaskBatchSubmissionResult:
    if remove_documents_batch_task is None:
        return TaskBatchSubmissionResult(
            queue_available=False,
            requested_unique_count=len(set(doc_ids)),
            enqueued_count=0,
        )

    return _submit_backpressure_limited_batch_request(
        task_submitter=remove_documents_batch_task,
        batch_name="document",
        items=doc_ids,
        pending_items=_get_pending_remove_doc_ids(),
    )


def enqueue_refresh_git(git_dir: str) -> bool:
    """Enqueue a refresh_git_repository task. Returns True if enqueued."""
    return submit_refresh_git_request(git_dir).enqueued


def submit_refresh_git_request(git_dir: str) -> TaskSubmissionResult:
    if refresh_git_repository_task is None or _huey is None:
        return TaskSubmissionResult(status="unavailable")
    git_dir_key = str(Path(git_dir).resolve())
    with _git_refresh_lock:
        if git_dir_key in _git_refresh_in_flight:
            logger.info("Skipping git refresh enqueue for %s because it is running", git_dir)
            return TaskSubmissionResult(status="already_pending")
    if is_backpressured(
        _huey,
        _task_backpressure_limit,
        item=git_dir,
        warning_message="Skipping git refresh enqueue for %s due to task queue backpressure (%d pending >= %d limit)",
    ):
        return TaskSubmissionResult(status="backpressured")
    enqueued = submit_single_task(
        refresh_git_repository_task,
        git_dir,
        task_kwargs={"priority": GIT_REFRESH_TASK_PRIORITY},
        pending_first_args=_get_pending_refresh_git_dirs(),
        pending_skip_log_message="Skipping git refresh enqueue for %s because a pending task already exists",
    )
    if enqueued:
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

    queue_item = project_override or "__global__"
    if is_backpressured(
        _huey,
        _task_backpressure_limit,
        item=queue_item,
        warning_message="Skipping rebuild enqueue for %s due to task queue backpressure (%d pending >= %d limit)",
    ):
        return TaskSubmissionResult(status="backpressured")

    rebuild_index_task(project_override, request_id=request_id)
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
    if is_backpressured(
        _huey,
        _task_backpressure_limit,
        item=f"reindex:{operation}",
        warning_message="Skipping reindex enqueue for %s due to task queue backpressure (%d pending >= %d limit)",
    ):
        return TaskSubmissionResult(status="backpressured")
    reindex_model_task(
        operation,
        model,
        truncate_dim,
        old_model,
        request_id=request_id,
        priority=REINDEX_TASK_PRIORITY,
    )
    return TaskSubmissionResult(status="enqueued")


def get_pending_task_count() -> int:
    return get_shared_pending_task_count(_huey)


def is_task_queue_available() -> bool:
    return _huey is not None and index_document_task is not None
