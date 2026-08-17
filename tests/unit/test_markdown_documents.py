"""Contract tests for Markdown planning and canonical document writing."""

import asyncio
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

import pytest

from searchkernel.api import (
    Cursor,
    IngestionFailureMode,
    IngestionReceipt,
    Record,
    RecordIdentity,
    RecordIngestionResult,
    get_chunker,
)

from mcp_markdown_ragdocs.indexing.markdown_documents import (
    MarkdownDocumentPlanner,
    SemanticDocumentWriter,
)
from tests.conftest import create_test_document


def _success_receipt() -> IngestionReceipt:
    return IngestionReceipt(
        source_kind="note",
        workspace_id=None,
        checkpoint=None,
        records=(
            RecordIngestionResult(
                source_kind="note",
                source_id="guide",
                workspace_id=None,
                status="committed",
            ),
        ),
    )


def _failure_receipt(error: str) -> IngestionReceipt:
    return IngestionReceipt(
        source_kind="note",
        workspace_id=None,
        checkpoint=None,
        records=(
            RecordIngestionResult(
                source_kind="note",
                source_id="guide",
                workspace_id=None,
                status="failed",
                error=error,
            ),
        ),
    )


class _Ingestor:
    def __init__(self, events: list[str], receipt: IngestionReceipt) -> None:
        self.events = events
        self.receipt = receipt

    async def index_records(
        self,
        records: Sequence[Record],
        *,
        checkpoint: Cursor | None = None,
        failure_mode: IngestionFailureMode = "strict",
    ) -> IngestionReceipt:
        del records, checkpoint, failure_mode
        self.events.append("index")
        return self.receipt


class _Storage:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.deleted: list[tuple[str, ...]] = []

    @property
    def db_manager(self) -> object:
        return object()

    def hydrate_record(self, identity: RecordIdentity | str) -> Record | None:
        del identity
        return None

    def hydrate_records(
        self,
        identities: Sequence[RecordIdentity],
    ) -> Mapping[str, Record | None]:
        del identities
        return {}

    def iter_records(self) -> Iterable[Record]:
        return ()

    def delete(self, storage_keys: Sequence[str]) -> None:
        self.events.append("delete")
        self.deleted.append(tuple(storage_keys))


def test_markdown_planner_preserves_identity_and_chunk_metadata(record_manager) -> None:
    """
    Plan a Markdown file with the manager's stable ID and metadata contract.
    """
    docs_dir = Path(record_manager._config.indexing.documents_path)
    file_path = create_test_document(
        docs_dir,
        "guide",
        "# Guide\n\nPlanner contract document",
    )
    planner = MarkdownDocumentPlanner(
        record_manager._config,
        [docs_dir],
        get_chunker(record_manager._config.chunking),
    )

    prepared = planner.plan(file_path)
    expected = record_manager.prepare_document(file_path)

    assert prepared.document.id == expected.document.id
    assert prepared.document.project_id == expected.document.project_id
    assert prepared.document.metadata == expected.document.metadata
    assert [record.storage_key for record in prepared.records] == [
        record.storage_key for record in expected.records
    ]
    assert prepared.records[0].metadata["doc_id"] == prepared.document.id
    assert prepared.records[0].metadata["file_path"] == file_path


def test_document_writer_indexes_before_deleting_stale_keys(record_manager) -> None:
    """
    Write planned records before removing stale storage memberships.
    """
    docs_dir = Path(record_manager._config.indexing.documents_path)
    file_path = create_test_document(docs_dir, "guide", "# Guide\n\nWriter contract")
    planner = MarkdownDocumentPlanner(
        record_manager._config,
        [docs_dir],
        get_chunker(record_manager._config.chunking),
    )
    prepared = planner.plan(file_path)
    events: list[str] = []
    storage = _Storage(events)
    writer = SemanticDocumentWriter(
        _Ingestor(events, _success_receipt()),
        storage,
    )

    new_keys = asyncio.run(
        writer.write(prepared, ("stale-key", *[record.storage_key for record in prepared.records]))
    )

    assert new_keys == tuple(record.storage_key for record in prepared.records)
    assert events == ["index", "delete"]
    assert storage.deleted == [("stale-key",)]


def test_document_writer_preserves_ingestion_failure_without_deleting_stale_keys(
    record_manager,
) -> None:
    """
    Surface an ingestion failure before mutating stale canonical records.
    """
    docs_dir = Path(record_manager._config.indexing.documents_path)
    file_path = create_test_document(docs_dir, "guide", "# Guide\n\nWriter failure")
    planner = MarkdownDocumentPlanner(
        record_manager._config,
        [docs_dir],
        get_chunker(record_manager._config.chunking),
    )
    prepared = planner.plan(file_path)
    events: list[str] = []
    storage = _Storage(events)
    writer = SemanticDocumentWriter(
        _Ingestor(events, _failure_receipt("embedding failed")),
        storage,
    )

    with pytest.raises(RuntimeError, match="embedding failed"):
        asyncio.run(writer.write(prepared, ("stale-key",)))

    assert events == ["index"]
    assert storage.deleted == []
