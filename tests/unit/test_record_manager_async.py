from datetime import UTC, datetime
from pathlib import Path

import pytest
from searchkernel.domain import Record, RecordStatus

import mcp_markdown_ragdocs.indexing.record_manager as record_manager_module
from tests.conftest import create_test_document


def _make_record(source_id: str, body: str) -> Record:
    timestamp = datetime(2026, 1, 1, tzinfo=UTC)
    return Record(
        source_kind="git_commit",
        source_id=source_id,
        title="Async record",
        body=body,
        created_at=timestamp,
        updated_at=timestamp,
        metadata={"author": "Test User"},
        status=RecordStatus.ACTIVE,
    )


@pytest.mark.asyncio
async def test_async_index_document_avoids_sync_bridge(
    record_manager,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Async document indexing stays on the caller's event-loop path."""
    document = create_test_document(
        tmp_path / "docs",
        "async-document",
        "# Async document\n\nIndexed without a sync bridge.",
    )

    def fail_if_called(_awaitable):
        raise AssertionError("async indexing must not use the sync bridge")

    monkeypatch.setattr(record_manager_module, "_run_async", fail_if_called)

    assert await record_manager.async_index_document(document, update_graph=False)
    assert await record_manager.async_index_document(
        document,
        force=True,
        update_graph=False,
    )


@pytest.mark.asyncio
async def test_async_index_records_avoids_sync_bridge(
    record_manager,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Async record batches use the ingestor without the sync bridge."""
    record = _make_record("async-record", "Indexed through the async manager.")

    def fail_if_called(_awaitable):
        raise AssertionError("async ingestion must not use the sync bridge")

    monkeypatch.setattr(record_manager_module, "_run_async", fail_if_called)

    assert await record_manager.async_index_record(record)
    assert await record_manager.async_index_records(())
    assert record_manager.storage.hydrate_record(record.storage_key) is not None
