"""Huey task definitions for indexing operations."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from src.indexing.bootstrap_checkpoint import mark_bootstrap_file_completed

if TYPE_CHECKING:
    from src.git.commit_indexer import CommitIndexer
    from huey import SqliteHuey

logger = logging.getLogger(__name__)


class IndexManagerLike(Protocol):
    """Structural type for objects that can index/remove documents."""

    def index_document(self, file_path: str, force: bool = False) -> None: ...
    def remove_document(self, doc_id: str) -> None: ...
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
remove_document_task = None
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
    global index_document_task, remove_document_task, refresh_git_repository_task
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
    remove_document_task = _remove_document
    refresh_git_repository_task = _refresh_git_repository
    logger.info("Indexing tasks registered with Huey")


def enqueue_index(file_path: str, force: bool = False) -> bool:
    """Enqueue an index_document task. Returns True if enqueued, False if no Huey."""
    if index_document_task is None or _huey is None:
        return False
    if get_pending_task_count() >= _task_backpressure_limit:
        logger.warning(
            "Skipping index enqueue for %s due to task queue backpressure (%d pending >= %d limit)",
            file_path,
            get_pending_task_count(),
            _task_backpressure_limit,
        )
        return False
    index_document_task(file_path, force=force)
    return True


def _get_pending_index_document_paths() -> set[str]:
    """Return file paths already pending in the queue for _index_document tasks."""
    if _huey is None:
        return set()

    try:
        pending_messages = _huey.storage.enqueued_items()
    except Exception:
        logger.warning(
            "Failed to inspect pending Huey tasks; startup batch dedupe disabled",
            exc_info=True,
        )
        return set()

    pending_paths: set[str] = set()
    for message in pending_messages:
        try:
            task = _huey.deserialize_task(message)
        except Exception:
            logger.warning(
                "Failed to deserialize pending Huey task while inspecting startup queue",
                exc_info=True,
            )
            continue

        if getattr(task, "name", None) != "_index_document":
            continue

        args = getattr(task, "args", ())
        if not args:
            continue

        file_path = args[0]
        if isinstance(file_path, str):
            pending_paths.add(file_path)

    return pending_paths


def enqueue_index_batch(file_paths: list[str], force: bool = False) -> int:
    """Enqueue many index tasks without watcher backpressure throttling.

    Intended for cold-start/bootstrap flows where the full corpus needs to be
    materialized durably by the worker.
    """
    if index_document_task is None or _huey is None:
        return 0

    pending_paths = set() if force else _get_pending_index_document_paths()
    enqueued = 0
    skipped_pending = 0
    seen_paths = set(pending_paths)
    for file_path in file_paths:
        if file_path in seen_paths:
            if file_path in pending_paths:
                skipped_pending += 1
            continue
        index_document_task(file_path, force=force)
        seen_paths.add(file_path)
        enqueued += 1

    if skipped_pending > 0:
        logger.info(
            "Skipped %d startup indexing task(s) already pending in queue",
            skipped_pending,
        )

    return enqueued


def get_pending_index_document_count(file_paths: list[str]) -> int:
    """Count how many of the given file paths are already pending in Huey."""
    if not file_paths:
        return 0

    pending_paths = _get_pending_index_document_paths()
    unique_paths = set(file_paths)
    return sum(1 for file_path in unique_paths if file_path in pending_paths)


def enqueue_remove(doc_id: str) -> bool:
    """Enqueue a remove_document task. Returns True if enqueued, False if no Huey."""
    if remove_document_task is None or _huey is None:
        return False
    if get_pending_task_count() >= _task_backpressure_limit:
        logger.warning(
            "Skipping remove enqueue for %s due to task queue backpressure (%d pending >= %d limit)",
            doc_id,
            get_pending_task_count(),
            _task_backpressure_limit,
        )
        return False
    remove_document_task(doc_id)
    return True


def enqueue_refresh_git(git_dir: str) -> bool:
    """Enqueue a refresh_git_repository task. Returns True if enqueued."""
    if refresh_git_repository_task is None or _huey is None:
        return False
    if get_pending_task_count() >= _task_backpressure_limit:
        logger.warning(
            "Skipping git refresh enqueue for %s due to task queue backpressure (%d pending >= %d limit)",
            git_dir,
            get_pending_task_count(),
            _task_backpressure_limit,
        )
        return False
    refresh_git_repository_task(git_dir)
    return True


def enqueue_refresh_git_batch(git_dirs: list[str]) -> int:
    """Enqueue many git refresh tasks without watcher backpressure throttling."""
    if refresh_git_repository_task is None or _huey is None:
        return 0

    enqueued = 0
    for git_dir in git_dirs:
        refresh_git_repository_task(git_dir)
        enqueued += 1
    return enqueued


def get_pending_task_count() -> int:
    if _huey is None:
        return 0
    return int(_huey.pending_count())


def is_task_queue_available() -> bool:
    return _huey is not None and index_document_task is not None
