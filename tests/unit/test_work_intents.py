from __future__ import annotations

from pathlib import Path

from huey import SqliteHuey

from mcp_markdown_ragdocs.coordination.work_intents import (
    FAILED,
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
    intent = store.submit("remove_document", str(tmp_path / "doc.md"), {"doc_id": "x"})
    assert intent.state == SUCCEEDED


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
