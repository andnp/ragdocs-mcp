from datetime import UTC, datetime

from searchkernel.domain import Record, RecordIdentity


def _record(source_id: str) -> Record:
    timestamp = datetime(2026, 1, 1, tzinfo=UTC)
    return Record(
        workspace_id="workspace",
        source_kind="note",
        source_id=source_id,
        title=source_id,
        body="canonical searchkernel record",
        created_at=timestamp,
        updated_at=timestamp,
    )


def test_record_identity_keeps_workspace_and_source_kind_in_storage_key() -> None:
    first = RecordIdentity("workspace-a", "note", "same-id")
    second = RecordIdentity("workspace-b", "note", "same-id")

    assert first.storage_key != second.storage_key
    assert RecordIdentity.from_storage_key(first.storage_key) == first


def test_record_manager_indexes_and_hydrates_canonical_records(record_manager) -> None:
    record = _record("record-1")

    assert record_manager.index_record(record) is True

    hydrated = record_manager.kernel.backend.hydrate_record(record.storage_key)
    assert hydrated is not None
    assert hydrated.body == record.body
    assert record_manager.keyword.search("canonical", 5)
