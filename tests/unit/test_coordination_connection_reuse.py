"""Connection lifecycle for TaskLeaseStore, WorkIntentStore, and huey's queue.

Covers the fd-churn fix: one sqlite3 connection reused per store instance
instead of one per call, safe under concurrent threads, closed on disposal,
and huey's queue.db connection sharing the same NORMAL durability policy as
the lease/intent rows in the same file.
"""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path

import pytest

from mcp_markdown_ragdocs.coordination.queue import build_queue_runtime
from mcp_markdown_ragdocs.coordination.task_leases import TaskLeaseStore
from mcp_markdown_ragdocs.coordination.work_intents import WorkIntentStore


class _ConnectCounter:
    """Wraps sqlite3.connect and counts invocations."""

    def __init__(self) -> None:
        self.count = 0
        self._real_connect = sqlite3.connect

    def __call__(self, *args: object, **kwargs: object) -> sqlite3.Connection:
        self.count += 1
        return self._real_connect(*args, **kwargs)  # type: ignore[arg-type]


class TestConnectionReuse:
    def test_task_lease_store_opens_one_connection_for_many_calls(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Repeated lease calls reuse the store's connection instead of reopening it.

        This is the churn this fix targets: heartbeats fire every 5s per
        active task, so a naive per-call `sqlite3.connect()` sustains a
        steady connect/close rate. Reusing one connection for the store's
        life means the connect count stays at 1 regardless of call volume.
        """
        counter = _ConnectCounter()
        monkeypatch.setattr(sqlite3, "connect", counter)

        store = TaskLeaseStore(tmp_path / "queue.db", timeout_seconds=30.0)
        opened_after_init = counter.count

        store.claim(
            "task-1", task_name="index_document", owner_token="owner-1",
            payload=b"payload", now=1.0,
        )
        for tick in range(2, 10):
            assert store.heartbeat("task-1", owner_token="owner-1", now=float(tick))
        store.complete("task-1", owner_token="owner-1", now=10.0)
        store.get("task-1")

        assert opened_after_init == 1
        assert counter.count == 1

    def test_work_intent_store_opens_one_connection_for_many_calls(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        counter = _ConnectCounter()
        monkeypatch.setattr(sqlite3, "connect", counter)

        store = WorkIntentStore(tmp_path / "queue.db")
        submitted = store.submit(
            "index_document", "docs/a.md", {"file_path": "docs/a.md"}, now=1.0
        )
        claimed = store.claim(submitted.intent_id, now=2.0)
        assert claimed is not None
        _, token = claimed
        assert store.start(submitted.intent_id, token)
        assert store.succeed(submitted.intent_id, token, now=3.0)
        store.get(submitted.intent_id)
        store.find("index_document", "docs/a.md")

        assert counter.count == 1


class TestThreadSafety:
    def test_task_lease_store_survives_concurrent_heartbeats(
        self, tmp_path: Path
    ) -> None:
        """The shared connection tolerates concurrent use from many threads.

        Mirrors the real shape: one lease store, one heartbeat thread per
        active task, all hitting the same store instance concurrently. A
        naive cached-connection-without-locking implementation raises
        sqlite3.ProgrammingError here ("SQLite objects created in a thread
        can only be used in that same thread").
        """
        store = TaskLeaseStore(tmp_path / "queue.db", timeout_seconds=30.0)
        task_count = 8
        heartbeats_per_task = 20
        for index in range(task_count):
            assert store.claim(
                f"task-{index}", task_name="index_document",
                owner_token=f"owner-{index}", payload=b"payload", now=0.0,
            )

        errors: list[BaseException] = []
        errors_lock = threading.Lock()
        barrier = threading.Barrier(task_count)

        def heartbeat_loop(task_index: int) -> None:
            barrier.wait()
            try:
                for _tick in range(heartbeats_per_task):
                    ok = store.heartbeat(
                        f"task-{task_index}", owner_token=f"owner-{task_index}", now=1.0
                    )
                    assert ok
            except BaseException as error:  # noqa: BLE001 - surfaced via errors list
                with errors_lock:
                    errors.append(error)

        threads = [
            threading.Thread(target=heartbeat_loop, args=(index,))
            for index in range(task_count)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=10.0)

        assert not errors, errors
        assert all(not thread.is_alive() for thread in threads)
        assert store.active_count(now=1.0) == task_count

    def test_work_intent_store_survives_concurrent_submit_and_claim(
        self, tmp_path: Path
    ) -> None:
        store = WorkIntentStore(tmp_path / "queue.db")
        worker_count = 8
        errors: list[BaseException] = []
        errors_lock = threading.Lock()
        barrier = threading.Barrier(worker_count)

        def submit_and_claim(worker_index: int) -> None:
            barrier.wait()
            try:
                intent = store.submit(
                    "index_document",
                    f"docs/{worker_index}.md",
                    {"file_path": f"docs/{worker_index}.md"},
                    now=1.0,
                )
                claimed = store.claim(intent.intent_id, now=2.0)
                assert claimed is not None
            except BaseException as error:  # noqa: BLE001
                with errors_lock:
                    errors.append(error)

        threads = [
            threading.Thread(target=submit_and_claim, args=(index,))
            for index in range(worker_count)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=10.0)

        assert not errors, errors
        assert len(store.list_active(limit=worker_count)) == worker_count


class TestConnectionDisposal:
    def test_task_lease_store_close_closes_the_connection(self, tmp_path: Path) -> None:
        store = TaskLeaseStore(tmp_path / "queue.db")
        store.close()

        with pytest.raises(sqlite3.ProgrammingError):
            store.get("task-1")

    def test_work_intent_store_close_closes_the_connection(self, tmp_path: Path) -> None:
        store = WorkIntentStore(tmp_path / "queue.db")
        store.close()

        with pytest.raises(sqlite3.ProgrammingError):
            store.get("intent-1")


class TestQueueDurabilityConsistency:
    def test_huey_connection_uses_normal_synchronous(self, tmp_path: Path) -> None:
        """huey's own connection matches the NORMAL durability of lease/intent rows.

        Without fsync=, huey never pragmas `synchronous` and stays at
        SQLite's compiled default FULL (2), inconsistent with the NORMAL (1)
        that TaskLeaseStore and WorkIntentStore already force on the same
        physical file.
        """
        runtime = build_queue_runtime(tmp_path / "queue.db")

        storage = runtime.huey.storage
        (mode,) = storage.conn.execute("PRAGMA synchronous").fetchone()  # type: ignore[attr-defined]

        assert mode == 1
