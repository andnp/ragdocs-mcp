from __future__ import annotations

from pathlib import Path

import pytest
from huey import SqliteHuey

from src.coordination.task_submission import (
    get_pending_task_count,
    get_pending_task_first_args,
    get_pending_task_values,
    is_backpressured,
    submit_single_task,
    submit_task_batch,
)


@pytest.fixture()
def huey_instance(tmp_path: Path) -> SqliteHuey:
    return SqliteHuey(
        name="test-task-submission",
        filename=str(tmp_path / "task-submission.db"),
        immediate=False,
    )


def test_get_pending_task_first_args_filters_matching_task_names(
    huey_instance: SqliteHuey,
) -> None:
    @huey_instance.task()
    def _index_document(file_path: str, force: bool = False) -> bool:
        return force or bool(file_path)

    @huey_instance.task()
    def _refresh_git_repository(git_dir: str) -> bool:
        return bool(git_dir)

    _index_document("/docs/a.md")
    _index_document("/docs/b.md", force=True)
    _refresh_git_repository("/repo/.git")

    pending_paths = get_pending_task_first_args(
        huey_instance,
        "_index_document",
        inspection_failure_log_message="inspect failed",
        deserialize_failure_log_message="deserialize failed",
    )

    assert pending_paths == {"/docs/a.md", "/docs/b.md"}


def test_get_pending_task_values_extracts_items_from_batch_args(
    huey_instance: SqliteHuey,
) -> None:
    @huey_instance.task()
    def _index_documents_batch(file_paths: list[str], force: bool = False) -> bool:
        return force or bool(file_paths)

    _index_documents_batch(["/docs/a.md", "/docs/b.md"])

    pending_paths = get_pending_task_values(
        huey_instance,
        {"_index_documents_batch"},
        value_extractor=lambda task: set(task.args[0]),
        inspection_failure_log_message="inspect failed",
        deserialize_failure_log_message="deserialize failed",
    )

    assert pending_paths == {"/docs/a.md", "/docs/b.md"}


def test_submit_single_task_skips_pending_first_arg() -> None:
    observed: list[tuple[str, dict[str, object]]] = []

    def _submit(file_path: str, **kwargs: object) -> None:
        observed.append((file_path, dict(kwargs)))

    enqueued = submit_single_task(
        _submit,
        "/docs/a.md",
        task_kwargs={"force": True},
        pending_first_args={"/docs/a.md"},
        pending_skip_log_message="skip %s",
    )

    assert enqueued is False
    assert observed == []


def test_submit_task_batch_coalesces_pending_and_duplicate_first_args() -> None:
    observed: list[tuple[str, dict[str, object]]] = []

    def _submit(file_path: str, **kwargs: object) -> None:
        observed.append((file_path, dict(kwargs)))

    enqueued = submit_task_batch(
        _submit,
        ["/docs/a.md", "/docs/a.md", "/docs/b.md", "/docs/c.md", "/docs/c.md"],
        task_kwargs={"force": False},
        pending_first_args={"/docs/a.md"},
        skipped_pending_log_message="skipped %d",
    )

    assert enqueued == 2
    assert observed == [
        ("/docs/b.md", {"force": False}),
        ("/docs/c.md", {"force": False}),
    ]


def test_is_backpressured_uses_pending_task_count(huey_instance: SqliteHuey) -> None:
    @huey_instance.task()
    def _index_document(file_path: str) -> bool:
        return bool(file_path)

    _index_document("/docs/a.md")

    assert get_pending_task_count(huey_instance) == 1
    assert is_backpressured(
        huey_instance,
        1,
        item="/docs/b.md",
        warning_message="Skipping index enqueue for %s due to task queue backpressure (%d pending >= %d limit)",
    ) is True
    assert is_backpressured(
        huey_instance,
        2,
        item="/docs/b.md",
        warning_message="Skipping index enqueue for %s due to task queue backpressure (%d pending >= %d limit)",
    ) is False