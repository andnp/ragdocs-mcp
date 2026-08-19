"""Provider-neutral contracts for the explicit indexing task runtime."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

from searchkernel.api import TaskBatchSubmissionResult, TaskSubmissionResult


class TaskSubmissionPort(Protocol):
    """Submission operations owned by one composed task runtime."""

    def submit_index_request(
        self, file_path: str, force: bool = False
    ) -> TaskSubmissionResult: ...

    def submit_index_request_batch(
        self, file_paths: list[str], force: bool = False
    ) -> TaskBatchSubmissionResult: ...

    def submit_index_batch(
        self,
        file_paths: list[str],
        force: bool = False,
        progressive: bool = False,
    ) -> TaskBatchSubmissionResult: ...

    def submit_record_batch(
        self, record_payloads: list[dict[str, object]]
    ) -> object | None: ...

    def submit_remove_request(self, doc_id: str) -> TaskSubmissionResult: ...

    def submit_remove_request_batch(
        self, doc_ids: list[str]
    ) -> TaskBatchSubmissionResult: ...

    def submit_refresh_git_request(self, git_dir: str) -> TaskSubmissionResult: ...

    def submit_refresh_git_batch(
        self, git_dirs: list[str]
    ) -> TaskBatchSubmissionResult: ...

    def submit_rebuild_request(
        self, project_override: str | None, *, request_id: str
    ) -> TaskSubmissionResult: ...

    def submit_reindex_request(
        self,
        operation: str,
        *,
        model: str | None,
        truncate_dim: int | None,
        old_model: str | None,
        request_id: str,
    ) -> TaskSubmissionResult: ...

    def get_pending_index_document_count(self, file_paths: list[str]) -> int: ...

    def get_pending_task_count(self) -> int: ...

    def is_task_queue_available(self) -> bool: ...


class TaskRuntimePort(TaskSubmissionPort, Protocol):
    """Submission plus task-backed lifecycle operations for one process."""

    def enqueue_index(self, file_path: str, force: bool = False) -> bool: ...

    def enqueue_index_batch(self, file_paths: list[str], force: bool = False) -> int: ...

    def enqueue_remove(self, doc_id: str) -> bool: ...

    def enqueue_refresh_git(self, git_dir: str) -> bool: ...

    def enqueue_refresh_git_batch(self, git_dirs: list[str]) -> int: ...

    def queue_identity(self) -> Path | None: ...


VectorStoreFactory = Callable[[Any, str], Any]


class ModelMigrationDependencies(Protocol):
    """Explicit dependencies needed to activate a migrated vector store."""

    config: Any
    build_vector_store: VectorStoreFactory
