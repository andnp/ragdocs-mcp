"""Huey task definitions for indexing operations."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Protocol

from src.coordination.task_submission import (
    coalesce_pending_first_args,
    get_pending_task_count as get_shared_pending_task_count,
    get_pending_task_first_args,
    get_pending_task_values,
    is_backpressured,
    submit_coalesced_batch_task,
    submit_single_task,
    submit_task_batch,
)
from src.indexing.bootstrap_checkpoint import (
    mark_bootstrap_file_completed,
    mark_bootstrap_files_completed,
)

if TYPE_CHECKING:
    from src.git.commit_indexer import CommitIndexer
    from huey import SqliteHuey

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TaskSubmissionResult:
    status: Literal["enqueued", "already_pending", "backpressured", "unavailable"]

    @property
    def accepted_by_queue(self) -> bool:
        return self.status in {"enqueued", "already_pending"}

    @property
    def should_retry_later(self) -> bool:
        return self.status == "backpressured"

    @property
    def queue_available(self) -> bool:
        return self.status != "unavailable"

    @property
    def enqueued(self) -> bool:
        return self.status == "enqueued"


@dataclass(frozen=True)
class TaskBatchSubmissionResult:
    queue_available: bool
    requested_unique_count: int
    enqueued_count: int
    already_pending_count: int = 0
    backpressured_items: tuple[str, ...] = ()

    @property
    def backpressured_count(self) -> int:
        return len(self.backpressured_items)

    @property
    def should_retry_later(self) -> bool:
        return bool(self.backpressured_items)

    @property
    def all_represented(self) -> bool:
        if not self.queue_available:
            return False
        if self.should_retry_later:
            return False
        return (
            self.enqueued_count + self.already_pending_count
            >= self.requested_unique_count
        )


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


# Module-level references set during initialization
_huey: SqliteHuey | None = None
_index_manager: IndexManagerLike | None = None
_commit_indexer: CommitIndexer | None = None
_task_backpressure_limit: int = 100
_bootstrap_index_path: Path | None = None
_bootstrap_documents_roots: list[Path] = []

# Task references (set after register_tasks is called)
index_document_task = None
index_documents_batch_task = None
remove_document_task = None
remove_documents_batch_task = None
refresh_git_repository_task = None


def register_tasks(
    huey: SqliteHuey,
    index_manager: IndexManagerLike,
    commit_indexer: CommitIndexer | None = None,
    task_backpressure_limit: int = 100,
    bootstrap_index_path: Path | None = None,
    bootstrap_documents_roots: list[Path] | None = None,
) -> None:
    """Register indexing tasks with the given Huey instance.

    Must be called before enqueuing tasks. Typically called during
    application startup when the worker is being configured.
    """
    global _huey, _index_manager, _commit_indexer, _task_backpressure_limit
    global _bootstrap_index_path, _bootstrap_documents_roots
    global index_document_task, index_documents_batch_task, remove_document_task
    global remove_documents_batch_task, refresh_git_repository_task
    _huey = huey
    _index_manager = index_manager
    _commit_indexer = commit_indexer
    _task_backpressure_limit = max(1, task_backpressure_limit)
    _bootstrap_index_path = bootstrap_index_path
    _bootstrap_documents_roots = list(bootstrap_documents_roots or [])

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
            logger.error("Task failed: index %s", file_path, exc_info=True)
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
                    logger.error(
                        "Task failed within batch: index %s",
                        file_path,
                        exc_info=True,
                    )

            if completed_paths:
                try:
                    _index_manager.persist()
                except Exception:
                    logger.error(
                        "Batch fallback persist failed for %d indexed document(s)",
                        len(completed_paths),
                        exc_info=True,
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
            logger.error("Task failed: remove %s", doc_id, exc_info=True)
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
                    logger.error(
                        "Task failed within batch: remove %s",
                        doc_id,
                        exc_info=True,
                    )

            if removed_doc_ids:
                try:
                    _index_manager.persist()
                except Exception:
                    logger.error(
                        "Batch fallback persist failed for %d removed document(s)",
                        len(removed_doc_ids),
                        exc_info=True,
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
        if _commit_indexer is None:
            logger.error("CommitIndexer not available for git refresh task")
            return False

        from src.git.parallel_indexer import (
            ParallelIndexingConfig,
            index_commits_parallel_sync,
        )
        from src.git.repository import get_commits_after_timestamp

        git_dir_path = Path(git_dir)

        try:
            last_timestamp = _commit_indexer.get_last_indexed_timestamp(str(git_dir_path))
            commit_hashes = get_commits_after_timestamp(git_dir_path, last_timestamp)
            if not commit_hashes:
                logger.info("Task completed: no new git commits for %s", git_dir_path)
                return True

            indexed = index_commits_parallel_sync(
                commit_hashes,
                git_dir_path,
                _commit_indexer,
                ParallelIndexingConfig(),
                200,
            )
            logger.info(
                "Task completed: refreshed git repository %s (%d commits)",
                git_dir_path,
                indexed,
            )
            return True
        except Exception:
            logger.error("Task failed: refresh git %s", git_dir_path, exc_info=True)
            return False

    index_document_task = _index_document
    index_documents_batch_task = _index_documents_batch
    remove_document_task = _remove_document
    remove_documents_batch_task = _remove_documents_batch
    refresh_git_repository_task = _refresh_git_repository
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
        pending_first_args=_get_pending_refresh_git_dirs(),
        pending_skip_log_message="Skipping git refresh enqueue for %s because a pending task already exists",
    )
    if enqueued:
        return TaskSubmissionResult(status="enqueued")
    return TaskSubmissionResult(status="already_pending")


def enqueue_refresh_git_batch(git_dirs: list[str]) -> int:
    """Enqueue many git refresh tasks without watcher backpressure throttling."""
    return submit_refresh_git_batch(git_dirs).enqueued_count


def submit_refresh_git_batch(git_dirs: list[str]) -> TaskBatchSubmissionResult:
    if refresh_git_repository_task is None or _huey is None:
        return TaskBatchSubmissionResult(
            queue_available=False,
            requested_unique_count=len(set(git_dirs)),
            enqueued_count=0,
        )

    pending_git_dirs = _get_pending_refresh_git_dirs()
    requested_unique_dirs = set(git_dirs)
    enqueued_count = submit_task_batch(
        refresh_git_repository_task,
        git_dirs,
        pending_first_args=pending_git_dirs,
        skipped_pending_log_message="Skipped %d startup git refresh task(s) already pending in queue",
    )
    already_pending_count = sum(
        1 for git_dir in requested_unique_dirs if git_dir in pending_git_dirs
    )
    return TaskBatchSubmissionResult(
        queue_available=True,
        requested_unique_count=len(requested_unique_dirs),
        enqueued_count=enqueued_count,
        already_pending_count=already_pending_count,
    )


def get_pending_task_count() -> int:
    return get_shared_pending_task_count(_huey)


def is_task_queue_available() -> bool:
    return _huey is not None and index_document_task is not None
