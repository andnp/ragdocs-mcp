"""Tests for describe_documents batching and first-hydrating-key semantics."""

from datetime import UTC, datetime

from searchkernel.domain import Record, RecordIdentity, RecordStatus


def _make_record(source_id: str, body: str) -> Record:
    timestamp = datetime(2026, 1, 1, tzinfo=UTC)
    return Record(
        source_kind="git_commit",
        source_id=source_id,
        title="Describe-documents record",
        body=body,
        created_at=timestamp,
        updated_at=timestamp,
        metadata={"file_path": f"/repo/{source_id}.md"},
        status=RecordStatus.ACTIVE,
    )


def test_describe_documents_reports_file_path_chunk_count_and_source_kind(
    record_manager,
) -> None:
    """Describe every live doc_id using its hydrated record's metadata."""
    first = _make_record("doc-one", "first body")
    second = _make_record("doc-two", "second body")
    assert record_manager.index_record(first) is True
    assert record_manager.index_record(second) is True

    descriptions = {
        description["doc_id"]: description
        for description in record_manager.describe_documents()
    }

    assert descriptions["doc-one"] == {
        "doc_id": "doc-one",
        "file_path": "/repo/doc-one.md",
        "chunk_count": 1,
        "source_kind": "git_commit",
    }
    assert descriptions["doc-two"]["file_path"] == "/repo/doc-two.md"


def test_describe_documents_uses_first_hydrating_key_in_list_order(
    record_manager,
) -> None:
    """When a doc_id's key list has a stale leading key, fall through to the next."""
    live = _make_record("doc-live", "live body")
    assert record_manager.index_record(live) is True

    stale_key = RecordIdentity(None, "git_commit", "doc-missing").storage_key
    record_manager._source_records["doc-with-stale-head"] = [
        stale_key,
        live.storage_key,
    ]

    descriptions = {
        description["doc_id"]: description
        for description in record_manager.describe_documents()
    }

    assert descriptions["doc-with-stale-head"]["file_path"] == "/repo/doc-live.md"
    assert descriptions["doc-with-stale-head"]["chunk_count"] == 2


def test_describe_documents_skips_doc_ids_whose_keys_all_miss(record_manager) -> None:
    """A doc_id whose every key fails to hydrate is omitted, not errored."""
    missing_key = RecordIdentity(None, "git_commit", "gone").storage_key
    record_manager._source_records["doc-all-missing"] = [missing_key]

    descriptions = {
        description["doc_id"]: description
        for description in record_manager.describe_documents()
    }

    assert "doc-all-missing" not in descriptions


def test_describe_documents_issues_a_bounded_number_of_storage_queries(
    record_manager,
    monkeypatch,
) -> None:
    """Hydration for many doc_ids happens in a handful of batched queries.

    Regression guard for the N+1 hydration bug: describe_documents used to call
    storage.hydrate_record once per doc_id (one locked SQL round trip each).
    """
    for index in range(50):
        assert record_manager.index_record(_make_record(f"doc-{index}", f"body {index}")) is True

    single_record_calls = 0
    batched_calls = 0
    original_hydrate_record = record_manager.storage.hydrate_record
    original_hydrate_records = record_manager.storage.hydrate_records

    def counting_hydrate_record(identity):
        nonlocal single_record_calls
        single_record_calls += 1
        return original_hydrate_record(identity)

    def counting_hydrate_records(identities):
        nonlocal batched_calls
        batched_calls += 1
        return original_hydrate_records(identities)

    monkeypatch.setattr(record_manager.storage, "hydrate_record", counting_hydrate_record)
    monkeypatch.setattr(record_manager.storage, "hydrate_records", counting_hydrate_records)

    descriptions = record_manager.describe_documents()

    assert len(descriptions) == 50
    assert single_record_calls == 0
    assert batched_calls <= 1
