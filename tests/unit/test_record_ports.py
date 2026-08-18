import types
from datetime import UTC, datetime

from searchkernel.domain import Record, RecordStatus


def _make_commit_record(source_id: str, body: str) -> Record:
    timestamp = datetime(2026, 1, 1, tzinfo=UTC)
    return Record(
        source_kind="git_commit",
        source_id=source_id,
        title="Fix login bug",
        body=body,
        created_at=timestamp,
        updated_at=timestamp,
        metadata={"author": "Test User", "files_changed": ["api.py"]},
        status=RecordStatus.ACTIVE,
    )


def test_iter_records_streams_lazily(record_manager) -> None:
    result = record_manager.storage.iter_records()

    assert isinstance(result, types.GeneratorType)


def test_iter_records_yields_every_stored_record(record_manager) -> None:
    records = [_make_commit_record(f"commit-{i}", f"message {i}") for i in range(5)]
    for record in records:
        assert record_manager.index_record(record) is True

    yielded_keys = {record.storage_key for record in record_manager.storage.iter_records()}

    assert yielded_keys == {record.storage_key for record in records}


def test_iter_records_hydrates_in_bounded_batches(record_manager, monkeypatch) -> None:
    records = [_make_commit_record(f"commit-{i}", f"message {i}") for i in range(5)]
    for record in records:
        assert record_manager.index_record(record) is True

    storage = record_manager.storage
    monkeypatch.setattr(storage.__class__, "_ITER_RECORDS_BATCH_SIZE", 2)
    original_hydrate_records = storage.hydrate_records
    batch_sizes: list[int] = []

    def _tracking_hydrate_records(identities):
        batch_sizes.append(len(identities))
        return original_hydrate_records(identities)

    monkeypatch.setattr(storage, "hydrate_records", _tracking_hydrate_records)

    consumed = list(storage.iter_records())

    assert len(consumed) == len(records)
    assert batch_sizes == [2, 2, 1]
