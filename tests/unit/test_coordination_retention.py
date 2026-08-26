"""Retention pruning for durable coordination tables.

Covers TaskLeaseStore.prune_terminal and WorkIntentStore.prune_terminal:
terminal rows past retention are removed, active/pending rows are never
touched regardless of age, rows inside the retention window survive, and
the delete is bounded per call.
"""

from __future__ import annotations

from pathlib import Path

from mcp_markdown_ragdocs.coordination.task_leases import TaskLeaseStore
from mcp_markdown_ragdocs.coordination.work_intents import WorkIntentStore


def _lease_store(tmp_path: Path) -> TaskLeaseStore:
    return TaskLeaseStore(tmp_path / "queue.db", timeout_seconds=30.0)


def _intent_store(tmp_path: Path) -> WorkIntentStore:
    return WorkIntentStore(tmp_path / "queue.db")


class TestTaskLeasePruneTerminal:
    def test_removes_completed_lease_past_retention(self, tmp_path: Path) -> None:
        store = _lease_store(tmp_path)
        store.claim(
            "task-old",
            task_name="index_document",
            owner_token="worker-1",
            payload=b"payload",
            now=0.0,
        )
        assert store.complete("task-old", owner_token="worker-1", now=0.0)

        pruned = store.prune_terminal(now=1000.0, retention_seconds=100.0)

        assert pruned == ["task-old"]
        assert store.get("task-old") is None

    def test_removes_failed_and_reclaimed_leases_past_retention(
        self, tmp_path: Path
    ) -> None:
        store = _lease_store(tmp_path)
        store.claim(
            "task-failed",
            task_name="index_document",
            owner_token="worker-1",
            payload=b"payload",
            now=0.0,
        )
        assert store.fail("task-failed", owner_token="worker-1", error="boom", now=0.0)

        store.claim(
            "task-reclaimed",
            task_name="index_document",
            owner_token="worker-2",
            payload=b"payload",
            now=0.0,
        )
        # heartbeat never refreshed, so it is well past the 30s lease timeout
        reclaimed = store.reclaim_expired(now=100.0)
        assert [lease.task_id for lease in reclaimed] == ["task-reclaimed"]

        pruned = store.prune_terminal(now=1000.0, retention_seconds=100.0)

        assert sorted(pruned) == ["task-failed", "task-reclaimed"]
        assert store.get("task-failed") is None
        assert store.get("task-reclaimed") is None

    def test_never_removes_active_lease_regardless_of_age(
        self, tmp_path: Path
    ) -> None:
        store = _lease_store(tmp_path)
        store.claim(
            "task-active",
            task_name="index_document",
            owner_token="worker-1",
            payload=b"payload",
            now=0.0,
        )

        pruned = store.prune_terminal(now=10_000_000.0, retention_seconds=1.0)

        assert pruned == []
        lease = store.get("task-active")
        assert lease is not None
        assert lease.state == "active"

    def test_keeps_terminal_lease_inside_retention_window(
        self, tmp_path: Path
    ) -> None:
        store = _lease_store(tmp_path)
        store.claim(
            "task-recent",
            task_name="index_document",
            owner_token="worker-1",
            payload=b"payload",
            now=0.0,
        )
        assert store.complete("task-recent", owner_token="worker-1", now=950.0)

        pruned = store.prune_terminal(now=1000.0, retention_seconds=100.0)

        assert pruned == []
        assert store.get("task-recent") is not None

    def test_prune_is_bounded_by_limit(self, tmp_path: Path) -> None:
        store = _lease_store(tmp_path)
        for index in range(5):
            task_id = f"task-{index}"
            store.claim(
                task_id,
                task_name="index_document",
                owner_token="worker-1",
                payload=b"payload",
                now=0.0,
            )
            assert store.complete(task_id, owner_token="worker-1", now=0.0)

        first_batch = store.prune_terminal(now=1000.0, retention_seconds=1.0, limit=2)
        assert len(first_batch) == 2

        remaining = sum(
            1 for index in range(5) if store.get(f"task-{index}") is not None
        )
        assert remaining == 3

        second_batch = store.prune_terminal(
            now=1000.0, retention_seconds=1.0, limit=10
        )
        assert len(second_batch) == 3
        assert all(store.get(f"task-{index}") is None for index in range(5))


class TestWorkIntentPruneTerminal:
    def _succeeded_intent(self, store: WorkIntentStore, key: str, *, at: float) -> str:
        submitted = store.submit("index_document", key, {"file_path": key}, now=0.0)
        claimed = store.claim(submitted.intent_id, now=0.0)
        assert claimed is not None
        intent, token = claimed
        assert store.start(intent.intent_id, token)
        assert store.succeed(intent.intent_id, token, now=at)
        return intent.intent_id

    def _failed_intent(self, store: WorkIntentStore, key: str, *, at: float) -> str:
        submitted = store.submit("index_document", key, {"file_path": key}, now=0.0)
        claimed = store.claim(submitted.intent_id, now=0.0)
        assert claimed is not None
        intent, token = claimed
        assert store.fail(intent.intent_id, token, "boom", now=at)
        return intent.intent_id

    def test_removes_succeeded_and_failed_intents_past_retention(
        self, tmp_path: Path
    ) -> None:
        store = _intent_store(tmp_path)
        succeeded_id = self._succeeded_intent(store, "docs/a.md", at=0.0)
        failed_id = self._failed_intent(store, "docs/b.md", at=0.0)

        pruned = store.prune_terminal(now=1000.0, retention_seconds=100.0)

        assert sorted(pruned) == sorted([succeeded_id, failed_id])
        assert store.get(succeeded_id) is None
        assert store.get(failed_id) is None

    def test_never_removes_pending_or_in_flight_intents_regardless_of_age(
        self, tmp_path: Path
    ) -> None:
        store = _intent_store(tmp_path)
        pending = store.submit(
            "index_document", "docs/pending.md", {"file_path": "docs/pending.md"}, now=0.0
        )
        claimed_submission = store.submit(
            "index_document", "docs/claimed.md", {"file_path": "docs/claimed.md"}, now=0.0
        )
        claim = store.claim(claimed_submission.intent_id, now=0.0)
        assert claim is not None

        pruned = store.prune_terminal(now=10_000_000.0, retention_seconds=1.0)

        assert pruned == []
        reloaded_pending = store.get(pending.intent_id)
        assert reloaded_pending is not None
        assert reloaded_pending.state == "pending"
        reloaded_claimed = store.get(claimed_submission.intent_id)
        assert reloaded_claimed is not None
        assert reloaded_claimed.state == "claimed"

    def test_keeps_terminal_intent_inside_retention_window(
        self, tmp_path: Path
    ) -> None:
        store = _intent_store(tmp_path)
        recent_id = self._succeeded_intent(store, "docs/recent.md", at=950.0)

        pruned = store.prune_terminal(now=1000.0, retention_seconds=100.0)

        assert pruned == []
        assert store.get(recent_id) is not None

    def test_prune_is_bounded_by_limit(self, tmp_path: Path) -> None:
        store = _intent_store(tmp_path)
        intent_ids = [
            self._succeeded_intent(store, f"docs/{index}.md", at=0.0)
            for index in range(5)
        ]

        first_batch = store.prune_terminal(now=1000.0, retention_seconds=1.0, limit=2)
        assert len(first_batch) == 2

        remaining = sum(
            1 for intent_id in intent_ids if store.get(intent_id) is not None
        )
        assert remaining == 3

        second_batch = store.prune_terminal(
            now=1000.0, retention_seconds=1.0, limit=10
        )
        assert len(second_batch) == 3
        assert all(store.get(intent_id) is None for intent_id in intent_ids)
