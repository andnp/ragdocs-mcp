"""Tests for instance-owned task runtime contracts."""

from __future__ import annotations

from pathlib import Path

from searchkernel.api import TaskBatchSubmissionResult, TaskSubmissionResult

from mcp_markdown_ragdocs.config import Config
from mcp_markdown_ragdocs.coordination.queue import build_queue_runtime
from mcp_markdown_ragdocs.coordination.task_submission import TaskSubmissionPort
from mcp_markdown_ragdocs.indexing.task_runtime import (
    RegisteredTaskFunction,
    RegisteredTaskHandle,
    TaskRuntime,
)


class _RegisteredHandle:
    def __init__(self, func: RegisteredTaskFunction) -> None:
        self.func = func

    def __call__(self, *args: object, **kwargs: object) -> object:
        return self.func(*args, **kwargs)


class _Submission:
    def submit_index_request(
        self, file_path: str, force: bool = False
    ) -> TaskSubmissionResult:
        return TaskSubmissionResult(status="enqueued")

    def submit_index_batch(
        self,
        file_paths: list[str],
        force: bool = False,
        progressive: bool = False,
    ) -> TaskBatchSubmissionResult:
        return TaskBatchSubmissionResult(
            queue_available=True,
            requested_unique_count=len(file_paths),
            enqueued_count=len(file_paths),
        )

    def submit_index_request_batch(
        self, file_paths: list[str], force: bool = False
    ) -> TaskBatchSubmissionResult:
        return self.submit_index_batch(file_paths, force=force)

    def submit_record_batch(
        self, record_payloads: list[dict[str, object]]
    ) -> object | None:
        return record_payloads

    def submit_remove_request(self, doc_id: str) -> TaskSubmissionResult:
        return TaskSubmissionResult(status="enqueued")

    def submit_remove_request_batch(
        self, doc_ids: list[str]
    ) -> TaskBatchSubmissionResult:
        return self.submit_index_batch(doc_ids)

    def submit_refresh_git_request(self, git_dir: str) -> TaskSubmissionResult:
        return TaskSubmissionResult(status="enqueued")

    def submit_refresh_git_batch(
        self, git_dirs: list[str]
    ) -> TaskBatchSubmissionResult:
        return self.submit_index_batch(git_dirs)

    def submit_rebuild_request(
        self, project_override: str | None, *, request_id: str
    ) -> TaskSubmissionResult:
        return TaskSubmissionResult(status="enqueued")

    def submit_reindex_request(
        self,
        operation: str,
        *,
        model: str | None,
        truncate_dim: int | None,
        old_model: str | None,
        request_id: str,
    ) -> TaskSubmissionResult:
        return TaskSubmissionResult(status="enqueued")


def test_task_runtime_keeps_queue_submission_and_handles_together(
    tmp_path: Path,
) -> None:
    """
    Given one composed queue and submission port.
    When a task runtime is created.
    Then it exposes the same instance-owned capabilities and handles.
    """
    submission = _Submission()
    assert isinstance(submission, TaskSubmissionPort)
    queue_runtime = build_queue_runtime(tmp_path / "queue.db")
    def _index_document(*args: object, **kwargs: object) -> object:
        return None

    handles: dict[str, RegisteredTaskHandle] = {
        "index_document": _RegisteredHandle(_index_document)
    }

    runtime = TaskRuntime(queue_runtime, submission, handles, config=Config())

    assert runtime.queue_runtime is queue_runtime
    assert runtime.submission is submission
    assert runtime.task_handles == handles
    assert runtime.gdrive_task_handles == {}
    assert runtime.submission.submit_index_request("note.md").status == "enqueued"


def test_task_runtime_instances_do_not_share_submission_or_queue(tmp_path: Path) -> None:
    """
    Given two independently composed runtimes.
    When each runtime receives distinct capabilities.
    Then queue and submission identity remains isolated.
    """
    first_submission = _Submission()
    second_submission = _Submission()
    first = TaskRuntime(
        build_queue_runtime(tmp_path / "first" / "queue.db"),
        first_submission,
        config=Config(),
    )
    second = TaskRuntime(
        build_queue_runtime(tmp_path / "second" / "queue.db"),
        second_submission,
        config=Config(),
    )

    assert first.queue_runtime is not second.queue_runtime
    assert first.queue_runtime.huey is not second.queue_runtime.huey
    assert first.submission is first_submission
    assert second.submission is second_submission
