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


def test_index_record_adds_record_to_canonical_stores(record_manager) -> None:
    record = _make_commit_record("abc123", "Fix the login bug in the API handler.")

    assert record_manager.index_record(record) is True

    hydrated = record_manager.storage.hydrate_record(record.storage_key)
    assert hydrated is not None
    assert hydrated.source_kind == "git_commit"
    assert record_manager.keyword.search("login", 5)
    assert record_manager.vector.search(
        record.embedding or [],
        5,
        model_name=record_manager.embedding_provider.model_name,
        dim=record_manager.embedding_provider.dim,
    )


def test_index_record_replaces_changed_record_content(record_manager) -> None:
    first = _make_commit_record("abc123", "Original message.")
    updated = _make_commit_record("abc123", "Updated message.")

    assert record_manager.index_record(first) is True
    assert record_manager.index_record(updated) is True

    hydrated = record_manager.storage.hydrate_record(updated.storage_key)
    assert hydrated is not None
    assert hydrated.body == "Updated message."
