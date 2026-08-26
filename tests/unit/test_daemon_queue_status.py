from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

from huey import SqliteHuey
from huey.utils import Error

from mcp_markdown_ragdocs.coordination.task_leases import TaskLeaseStore
from mcp_markdown_ragdocs.coordination.queue import build_queue_runtime
from mcp_markdown_ragdocs.daemon.admin_payloads import _build_queue_status_payload
from mcp_markdown_ragdocs.daemon.queue_status import (
    QueueFailure,
    get_queue_stats,
    purge_queue_state,
)
from mcp_markdown_ragdocs.indexing.git_refresh_state import save_progress


def test_get_queue_stats_includes_pending_and_scheduled_task_details(
    tmp_path: Path,
) -> None:
    """Queue stats expose inspectable pending and scheduled task summaries.

    This verifies the admin payload keeps its aggregate counts while adding
    decoded task metadata operators can inspect directly.
    """

    db_path = tmp_path / "queue.db"
    huey = SqliteHuey(name="queue-status", filename=str(db_path), immediate=False)

    @huey.task(priority=4)
    def pending_task(file_path: str) -> str:
        return file_path

    @huey.task(priority=9)
    def scheduled_task(repo_path: str) -> str:
        return repo_path

    pending_result = pending_task("docs/guide.md")
    scheduled_signature = scheduled_task.s("/repo")
    scheduled_eta = datetime.now(UTC) + timedelta(minutes=5)
    scheduled_signature.eta = scheduled_eta
    huey.storage.add_to_schedule(
        huey.serialize_task(scheduled_signature),
        scheduled_eta,
    )

    stats = get_queue_stats(
        huey,
        worker_running=True,
        backpressure_limit=10,
    )
    payload = stats.to_dict()

    assert stats.pending_count == 1
    assert stats.scheduled_count == 1
    assert stats.historical_failure_count == 0
    assert payload["task_counts"] == {
        "pending_task": 1,
        "scheduled_task": 1,
    }
    assert payload["pending_tasks"] == [
        {
            "task_id": pending_result.id,
            "task_name": "pending_task",
            "state": "pending",
            "source_queue": "pending",
            "eta": None,
            "priority": 4,
            "retries": 0,
        }
    ]
    assert payload["scheduled_tasks"] == [
        {
            "task_id": scheduled_signature.id,
            "task_name": "scheduled_task",
            "state": "scheduled",
            "source_queue": "scheduled",
            "eta": scheduled_eta.isoformat(),
            "priority": 9,
            "retries": 0,
        }
    ]


def test_purge_queue_state_clears_only_selected_state(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "queue.db"
    huey = SqliteHuey(name="queue-purge", filename=str(db_path), immediate=False)

    @huey.task()
    def pending_task(file_path: str) -> str:
        return file_path

    @huey.task()
    def scheduled_task(repo_path: str) -> str:
        return repo_path

    pending_task("docs/guide.md")
    scheduled_signature = scheduled_task.s("/repo")
    scheduled_eta = datetime.now(UTC) + timedelta(minutes=5)
    scheduled_signature.eta = scheduled_eta
    huey.storage.add_to_schedule(
        huey.serialize_task(scheduled_signature),
        scheduled_eta,
    )
    huey.storage.put_data(
        "failure-1",
        huey.serializer.serialize(
            Error(
                {
                    "task_id": "failure-1",
                    "task_name": "mcp_markdown_ragdocs.indexing.tasks.delete_document",
                    "error": "boom",
                    "retries": 1,
                }
            )
        ),
        is_result=True,
    )

    payload = purge_queue_state(
        huey,
        state="pending",
        worker_running=True,
        backpressure_limit=10,
    ).to_dict()

    assert payload["purged_state"] == "pending"
    assert payload["purged_counts"] == {
        "pending": 1,
        "scheduled": 0,
        "failed": 0,
    }
    assert payload["pending_count"] == 0
    assert payload["scheduled_count"] == 1
    assert payload["failed_count"] == 1
    assert payload["historical_failure_count"] == 1


def test_get_queue_stats_reports_active_lease_count(tmp_path: Path) -> None:
    db_path = tmp_path / "queue.db"
    huey = SqliteHuey(name="queue-status", filename=str(db_path), immediate=False)
    lease_store = TaskLeaseStore(db_path)
    assert lease_store.claim(
        "running-1",
        task_name="index_document",
        owner_token="worker-1",
        payload=b"serialized-task",
    )

    stats = get_queue_stats(huey)

    assert stats.running_count == 1


def test_queue_status_payload_includes_git_refresh_progress(tmp_path: Path) -> None:
    db_path = tmp_path / "queue.db"
    huey = SqliteHuey(name="queue-progress", filename=str(db_path), immediate=False)
    save_progress(
        tmp_path,
        tmp_path / "repo" / ".git",
        {
            "state": "running",
            "processed_count": 2,
            "discovered_count": 5,
        },
    )

    payload = _build_queue_status_payload(
        queue_runtime=build_queue_runtime(db_path),
        worker_running=True,
    )

    assert payload["git_refresh_progress"] == [
        {
            "repository_path": str((tmp_path / "repo" / ".git").resolve()),
            "state": "running",
            "processed_count": 2,
            "discovered_count": 5,
        }
    ]
    huey.storage.close()


def _put_failure(
    huey: SqliteHuey,
    *,
    task_id: str,
    task_name: str = "mcp_markdown_ragdocs.indexing.tasks.index_document",
    error: str = "boom",
    retries: int = 1,
) -> None:
    huey.storage.put_data(
        task_id,
        huey.serializer.serialize(
            Error(
                {
                    "task_id": task_id,
                    "task_name": task_name,
                    "error": error,
                    "retries": retries,
                }
            )
        ),
        is_result=True,
    )


def test_get_queue_stats_reports_no_failures_for_empty_result_store(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "queue.db"
    huey = SqliteHuey(name="queue-no-failures", filename=str(db_path), immediate=False)

    stats = get_queue_stats(huey)

    assert stats.recent_failures == []
    assert stats.historical_failure_count == 0
    assert stats.failed_count == 0


def test_get_queue_stats_decodes_several_failures(tmp_path: Path) -> None:
    db_path = tmp_path / "queue.db"
    huey = SqliteHuey(name="queue-several-failures", filename=str(db_path), immediate=False)

    _put_failure(huey, task_id="failure-1", error="boom-1", retries=1)
    _put_failure(huey, task_id="failure-2", error="boom-2", retries=2)
    _put_failure(huey, task_id="failure-3", error="boom-3", retries=0)

    stats = get_queue_stats(huey)

    assert stats.historical_failure_count == 3
    assert sorted(stats.recent_failures, key=lambda f: f.task_id) == [
        QueueFailure(
            task_id="failure-1",
            task_name="index_document",
            error="boom-1",
            retries=1,
        ),
        QueueFailure(
            task_id="failure-2",
            task_name="index_document",
            error="boom-2",
            retries=2,
        ),
        QueueFailure(
            task_id="failure-3",
            task_name="index_document",
            error="boom-3",
            retries=0,
        ),
    ]


def test_get_queue_stats_skips_undecodable_result_row(tmp_path: Path) -> None:
    """A row whose value cannot be deserialized as an Error is dropped, not raised."""
    db_path = tmp_path / "queue.db"
    huey = SqliteHuey(name="queue-bad-row", filename=str(db_path), immediate=False)

    _put_failure(huey, task_id="failure-good", error="boom")
    huey.storage.put_data("garbage-key", b"not a valid serialized payload", is_result=True)

    stats = get_queue_stats(huey)

    assert stats.historical_failure_count == 1
    assert [f.task_id for f in stats.recent_failures] == ["failure-good"]


def test_collect_failures_does_not_requery_storage_per_row(tmp_path: Path) -> None:
    """Regression guard: failure decoding must not issue one storage query per row.

    Before the fix, decoding failures called huey.storage.peek_data(task_id) once for
    every row returned by result_items(), even though result_items() already performed
    a single bulk query that returned every row's value. Against the pre-fix
    implementation this assertion fails because peek_data is called once per row
    (verified separately against the old code: 25 rows -> 25 peek_data calls).
    """
    db_path = tmp_path / "queue.db"
    huey = SqliteHuey(name="queue-n-plus-one", filename=str(db_path), immediate=False)

    failure_count = 25
    for index in range(failure_count):
        _put_failure(huey, task_id=f"failure-{index}")

    peek_calls = {"count": 0}
    original_peek_data = huey.storage.peek_data

    def counting_peek_data(key: object) -> object:
        peek_calls["count"] += 1
        return original_peek_data(key)

    huey.storage.peek_data = counting_peek_data  # type: ignore[method-assign]

    stats = get_queue_stats(huey)

    assert stats.historical_failure_count == failure_count
    assert peek_calls["count"] == 0
