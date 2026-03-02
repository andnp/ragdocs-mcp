"""Huey task definitions for indexing operations."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from huey import SqliteHuey

logger = logging.getLogger(__name__)


class IndexManagerLike(Protocol):
    """Structural type for objects that can index/remove documents."""

    def index_document(self, file_path: str, force: bool = False) -> None: ...
    def remove_document(self, doc_id: str) -> None: ...


# Module-level references set during initialization
_huey: SqliteHuey | None = None
_index_manager: IndexManagerLike | None = None

# Task references (set after register_tasks is called)
index_document_task = None
remove_document_task = None


def register_tasks(huey: SqliteHuey, index_manager: IndexManagerLike) -> None:
    """Register indexing tasks with the given Huey instance.

    Must be called before enqueuing tasks. Typically called during
    application startup when the worker is being configured.
    """
    global _huey, _index_manager, index_document_task, remove_document_task
    _huey = huey
    _index_manager = index_manager

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

    index_document_task = _index_document
    remove_document_task = _remove_document
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
