from __future__ import annotations

import sqlite3
from pathlib import Path

from huey import SqliteHuey

from mcp_markdown_ragdocs.coordination.work_intents import (
    FAILED,
    PENDING,
    SUCCEEDED,
    WorkIntentStore,
)
from mcp_markdown_ragdocs.indexing import tasks


class _DeterministicIndexManager:
    def __init__(self, *, fail_once: bool = False) -> None:
        self.ingestor = object()
        self.fail_once = fail_once
        self.indexed: list[str] = []
        self.removed: list[str] = []

    def index_document(self, file_path: str, force: bool = False) -> bool:
        del force
        if self.fail_once:
            self.fail_once = False
            raise RuntimeError("deterministic indexing failure")
        self.indexed.append(file_path)
        return True

    def index_documents(
        self,
        file_paths: list[str],
        force: bool = False,
        persist: bool = False,
    ) -> None:
        for file_path in file_paths:
            self.index_document(file_path, force=force)
        del persist

    def remove_document(self, doc_id: str) -> None:
        self.removed.append(doc_id)

    def remove_documents(self, doc_ids: list[str], persist: bool = False) -> None:
        self.removed.extend(doc_ids)
        del persist

    def persist(self) -> None:
        return

    def index_record(self, record) -> None:
        del record


def _register(
    tmp_path: Path,
    manager: _DeterministicIndexManager,
) -> SqliteHuey:
    huey = SqliteHuey(
        name="intent-tests",
        filename=str(tmp_path / "queue.db"),
        immediate=False,
    )
    tasks.register_tasks(huey, manager)
    return huey


def _execute_one(huey: SqliteHuey) -> None:
    task = huey.dequeue()
    assert task is not None
    huey.execute(task)


def test_duplicate_remove_submissions_coalesce_canonical_identity(tmp_path: Path) -> None:
    manager = _DeterministicIndexManager()
    huey = _register(tmp_path, manager)

    first = tasks.submit_remove_request(str(tmp_path / "docs" / ".." / "doc.md"))
    second = tasks.submit_remove_request(str(tmp_path / "doc.md"))

    assert first.status == "enqueued"
    assert second.status == "already_pending"
    assert huey.pending_count() == 1

    _execute_one(huey)

    store = WorkIntentStore(tmp_path / "queue.db")
    active = store.list_active()
    assert active == []
    completed = store.find("remove_document", str(tmp_path / "doc.md"))
    assert completed is not None
    assert completed.state == SUCCEEDED
    reopened = store.submit(
        "remove_document",
        str(tmp_path / "doc.md"),
        {"doc_id": "x"},
    )
    assert reopened.state == PENDING


def test_stale_claim_cannot_terminalize_re_pended_intent(tmp_path: Path) -> None:
    store = WorkIntentStore(tmp_path / "queue.db", claim_timeout_seconds=10)
    original = store.submit("index_document", "doc.md", {"file_path": "doc.md"}, now=1)
    old_claim = store.claim(original.intent_id, now=2)
    assert old_claim is not None

    assert store.recover_stale_claims(now=20) == 1
    new_claim = store.claim(original.intent_id, now=21)
    assert new_claim is not None
    assert not store.succeed(original.intent_id, old_claim[1], now=22)
    assert store.succeed(original.intent_id, new_claim[1], now=23)
    completed = store.get(original.intent_id)
    assert completed is not None
    assert completed.state == SUCCEEDED


def test_failed_indexing_reopens_after_worker_restart(tmp_path: Path) -> None:
    manager = _DeterministicIndexManager(fail_once=True)
    huey = _register(tmp_path, manager)

    first = tasks.submit_index_request(str(tmp_path / "doc.md"))
    assert first.status == "enqueued"
    _execute_one(huey)

    store = WorkIntentStore(tmp_path / "queue.db")
    failed = store.list_active()
    assert failed == []
    intent = store.find("index_document", str(tmp_path / "doc.md"))
    assert intent is not None
    assert intent.state == FAILED

    retry = tasks.submit_index_request(str(tmp_path / "doc.md"))
    assert retry.status == "enqueued"
    _execute_one(huey)

    recovered = store.find("index_document", str(tmp_path / "doc.md"))
    assert recovered is not None
    assert recovered.state == SUCCEEDED
    assert manager.indexed == [str(tmp_path / "doc.md")]


def test_completed_indexing_reopens_for_later_update(tmp_path: Path) -> None:
    manager = _DeterministicIndexManager()
    huey = _register(tmp_path, manager)
    file_path = str(tmp_path / "doc.md")

    assert tasks.submit_index_request(file_path).status == "enqueued"
    _execute_one(huey)
    assert tasks.submit_index_request(file_path).status == "enqueued"
    _execute_one(huey)

    assert manager.indexed == [file_path, file_path]


def test_claim_release_reopens_intent_but_active_claim_is_exclusive(
    tmp_path: Path,
) -> None:
    store = WorkIntentStore(tmp_path / "queue.db", claim_timeout_seconds=10)
    intent = store.submit("index_document", "doc.md", {"file_path": "doc.md"}, now=1)

    first = store.claim(intent.intent_id, now=2)
    assert first is not None
    assert store.claim(intent.intent_id, now=3) is None
    assert store.succeed(intent.intent_id, "wrong-token", now=4) is False
    assert store.release(intent.intent_id, first[1], now=5) is True

    reopened = store.claim(intent.intent_id, now=6)
    assert reopened is not None
    assert reopened[0].attempt == 2


def test_reclaim_stale_claim_replaces_token(tmp_path: Path) -> None:
    store = WorkIntentStore(tmp_path / "queue.db", claim_timeout_seconds=10)
    intent = store.submit("index_document", "doc.md", {"file_path": "doc.md"}, now=1)
    claim = store.claim(intent.intent_id, now=2)
    assert claim is not None

    reclaimed = store.reclaim_stale_claim(intent.intent_id, claim[1], now=20)

    assert reclaimed is not None
    assert reclaimed[1] != claim[1]
    assert reclaimed[0].state == "claimed"
    assert not store.succeed(intent.intent_id, claim[1], now=21)
    assert store.succeed(intent.intent_id, reclaimed[1], now=22)


def test_force_reopen_preserves_active_claim(tmp_path: Path) -> None:
    store = WorkIntentStore(tmp_path / "queue.db", claim_timeout_seconds=10)
    intent = store.submit("index_document", "doc.md", {"file_path": "old"}, now=1)
    claim = store.claim(intent.intent_id, now=2)
    assert claim is not None

    submitted = store.submit(
        "index_document",
        "doc.md",
        {"file_path": "new"},
        force_reopen=True,
        now=3,
    )

    assert submitted.state == "claimed"
    assert submitted.claim_token == claim[1]
    assert submitted.payload == {"file_path": "old"}
    assert store.start(intent.intent_id, claim[1])


def test_force_reopen_allows_stale_claim(tmp_path: Path) -> None:
    store = WorkIntentStore(tmp_path / "queue.db", claim_timeout_seconds=10)
    intent = store.submit("index_document", "doc.md", {"file_path": "old"}, now=1)
    claim = store.claim(intent.intent_id, now=2)
    assert claim is not None

    submitted = store.submit(
        "index_document",
        "doc.md",
        {"file_path": "new"},
        force_reopen=True,
        now=20,
    )

    assert submitted.state == PENDING
    assert submitted.claim_token is None
    assert submitted.payload == {"file_path": "new"}


def test_index_document_automatic_retries_stop_at_three_and_force_reopens(
    tmp_path: Path,
) -> None:
    store = WorkIntentStore(tmp_path / "queue.db")
    intent = store.submit("index_document", "doc.md", {"file_path": "doc.md"})

    for failure_number in range(1, 4):
        claim = store.claim(intent.intent_id)
        assert claim is not None
        assert store.fail(intent.intent_id, claim[1], f"failure {failure_number}")
        submitted = store.submit(
            "index_document",
            "doc.md",
            {"file_path": "doc.md"},
        )
        assert submitted.state == (PENDING if failure_number < 3 else FAILED)

    terminal = store.get(intent.intent_id)
    assert terminal is not None
    assert terminal.failure_count == 3
    assert store.claim(intent.intent_id) is None

    forced = store.submit(
        "index_document",
        "doc.md",
        {"file_path": "doc.md"},
        force_reopen=True,
    )
    assert forced.state == PENDING
    forced_claim = store.claim(intent.intent_id)
    assert forced_claim is not None
    assert store.succeed(intent.intent_id, forced_claim[1])
    recovered = store.get(intent.intent_id)
    assert recovered is not None
    assert recovered.failure_count == 0


def test_work_intent_failure_count_is_added_to_legacy_database(tmp_path: Path) -> None:
    database = tmp_path / "queue.db"
    with sqlite3.connect(database) as connection:
        connection.execute(
            """
            CREATE TABLE work_intents (
                intent_id TEXT PRIMARY KEY,
                operation TEXT NOT NULL,
                canonical_key TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                state TEXT NOT NULL,
                claim_token TEXT,
                claim_observed_at REAL,
                observed_at REAL NOT NULL,
                attempt INTEGER NOT NULL,
                error TEXT,
                UNIQUE(operation, canonical_key)
            )
            """
        )
        connection.execute(
            """INSERT INTO work_intents VALUES
            ('legacy', 'index_document', 'doc.md', '{}', 'failed', NULL,
             NULL, 1, 1, 'old failure')"""
        )

    intent = WorkIntentStore(database).find("index_document", "doc.md")

    assert intent is not None
    assert intent.failure_count == 0
