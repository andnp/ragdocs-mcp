from datetime import UTC, datetime

import pytest
from searchkernel.domain import Record, RecordIdentity
from searchkernel.indexing.async_ingestion import AsyncIndexIngestor
from searchkernel.indices.local import LocalRecordBackend, LocalVectorStore


def _record(
    source_id: str,
    *,
    workspace_id: str | None = None,
    embedding: list[float] | None = None,
) -> Record:
    now = datetime.now(UTC)
    return Record(
        workspace_id=workspace_id,
        source_kind="note",
        source_id=source_id,
        title=source_id,
        body="canonical searchkernel record",
        created_at=now,
        updated_at=now,
        embedding=embedding,
        embedding_model="test-model" if embedding is not None else None,
    )


def test_record_identity_keeps_workspace_and_source_kind_in_storage_key():
    first = RecordIdentity("workspace-a", "note", "same-id")
    second = RecordIdentity("workspace-b", "note", "same-id")

    assert first.storage_key != second.storage_key
    assert RecordIdentity.from_storage_key(first.storage_key) == first


def test_local_vector_store_enforces_model_dimension(tmp_path):
    backend = LocalRecordBackend(tmp_path / "records.db")
    store = LocalVectorStore(backend)
    record = _record("note-1", embedding=[1.0, 0.0])

    store.upsert([record], "test-model", 2)

    with pytest.raises(ValueError, match="dimension mismatch"):
        store.search([1.0], 1, model_name="test-model", dim=2)


@pytest.mark.asyncio
async def test_async_ingestor_reports_lenient_failures_and_commits_successes():
    class Indexer:
        def index_record(self, record: Record) -> bool:
            if record.source_id == "bad":
                raise RuntimeError("broken record")
            return True

    receipt = await AsyncIndexIngestor(Indexer()).index_records(
        [_record("good"), _record("bad"), _record("later")],
        checkpoint="cursor-1",
        failure_mode="lenient",
    )

    assert receipt.checkpoint == "cursor-1"
    assert receipt.committed == 2
    assert receipt.failed == 1
    assert receipt.failures[0].source_id == "bad"


@pytest.mark.asyncio
async def test_async_ingestor_strict_mode_stops_after_first_failure():
    class Indexer:
        def index_record(self, record: Record) -> bool:
            if record.source_id == "bad":
                raise RuntimeError("broken record")
            return True

    receipt = await AsyncIndexIngestor(Indexer()).index_records(
        [_record("good"), _record("bad"), _record("later")],
        failure_mode="strict",
    )

    assert [result.source_id for result in receipt.records] == ["good", "bad"]
    assert receipt.committed == 1
    assert receipt.failed == 1
