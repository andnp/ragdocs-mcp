"""Huey task definitions for indexing operations."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from src.git.commit_indexer import CommitIndexer
    from huey import SqliteHuey

logger = logging.getLogger(__name__)


class IndexManagerLike(Protocol):
    """Structural type for objects that can index/remove documents."""

    def index_document(self, file_path: str, force: bool = False) -> None: ...
    def remove_document(self, doc_id: str) -> None: ...


# Module-level references set during initialization
_huey: SqliteHuey | None = None
_index_manager: IndexManagerLike | None = None
_commit_indexer: CommitIndexer | None = None

# Task references (set after register_tasks is called)
index_document_task = None
remove_document_task = None
refresh_git_repository_task = None


def register_tasks(
    huey: SqliteHuey,
    index_manager: IndexManagerLike,
    commit_indexer: CommitIndexer | None = None,
) -> None:
    """Register indexing tasks with the given Huey instance.

    Must be called before enqueuing tasks. Typically called during
    application startup when the worker is being configured.
    """
    global _huey, _index_manager, _commit_indexer
    global index_document_task, remove_document_task, refresh_git_repository_task
    _huey = huey
    _index_manager = index_manager
    _commit_indexer = commit_indexer

    @huey.task()
    def _index_document(file_path: str, force: bool = False) -> bool:
        """Index or re-index a single document."""
        if _index_manager is None:
            logger.error("IndexManager not available for task execution")
            return False
        try:
            _index_manager.index_document(file_path, force=force)
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
    if index_document_task is None:
        return False
    index_document_task(file_path, force=force)
    return True


def enqueue_remove(doc_id: str) -> bool:
    """Enqueue a remove_document task. Returns True if enqueued, False if no Huey."""
    if remove_document_task is None:
        return False
    remove_document_task(doc_id)
    return True


def enqueue_refresh_git(git_dir: str) -> bool:
    """Enqueue a refresh_git_repository task. Returns True if enqueued."""
    if refresh_git_repository_task is None:
        return False
    refresh_git_repository_task(git_dir)
    return True
