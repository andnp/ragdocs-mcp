from __future__ import annotations

from pathlib import Path

from mcp_markdown_ragdocs.coordination.queue import build_queue_runtime
from mcp_markdown_ragdocs.coordination.task_leases import TaskLeaseStore
from mcp_markdown_ragdocs.coordination.work_intents import WorkIntentStore


def test_work_intent_store_preserves_durable_state_across_reopen(
    tmp_path: Path,
) -> None:
    """
    Preserve a submitted and claimed work intent when the store is reopened.
    """
    database = tmp_path / "coordination.db"
    store = WorkIntentStore(database)
    submitted = store.submit(
        "index_document",
        "docs/guide.md",
        {"file_path": "docs/guide.md"},
        now=10.0,
    )
    claimed = store.claim(submitted.intent_id, now=11.0)

    assert claimed is not None

    reopened = WorkIntentStore(database).get(submitted.intent_id)

    assert reopened is not None
    assert reopened.intent_id == submitted.intent_id
    assert reopened.operation == "index_document"
    assert reopened.canonical_key == "docs/guide.md"
    assert reopened.payload == {"file_path": "docs/guide.md"}
    assert reopened.state == "claimed"
    assert reopened.claim_token == claimed[1]
    assert reopened.attempt == 1


def test_task_lease_store_preserves_durable_lease_across_reopen(
    tmp_path: Path,
) -> None:
    """
    Preserve active task ownership and replay data when the lease store reopens.
    """
    database = tmp_path / "coordination.db"
    store = TaskLeaseStore(database, timeout_seconds=30.0)

    assert store.claim(
        "task-1",
        task_name="index_document",
        owner_token="worker-1",
        payload=b"serialized-task",
        now=20.0,
    )

    reopened = TaskLeaseStore(database, timeout_seconds=30.0).get("task-1")

    assert reopened is not None
    assert reopened.task_id == "task-1"
    assert reopened.task_name == "index_document"
    assert reopened.state == "active"
    assert reopened.owner_token == "worker-1"
    assert reopened.payload == b"serialized-task"
    assert reopened.attempt == 1


def test_queue_runtime_preserves_huey_task_visibility_across_reopen(
    tmp_path: Path,
) -> None:
    """
    Keep an enqueued Huey task visible through a reopened queue runtime.
    """
    database = tmp_path / "queue.db"
    runtime = build_queue_runtime(database)

    @runtime.huey.task()
    def record_value(value: str) -> str:
        return value

    record_value("queued")

    reopened = build_queue_runtime(database)

    assert runtime.db_path == database
    assert runtime.huey.pending_count() == 1
    assert reopened.huey.pending_count() == 1
