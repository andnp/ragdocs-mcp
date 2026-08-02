from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

from huey import SqliteHuey
from huey.utils import Error

from mcp_markdown_ragdocs.coordination.task_leases import TaskLeaseStore
from mcp_markdown_ragdocs.daemon.queue_status import get_queue_stats, purge_queue_state


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
