"""Contract tests for Markdown planning and canonical document writing."""

import asyncio
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

import pytest

from searchkernel.api import (
    ContentSource,
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
from mcp_markdown_ragdocs.indexing.record_manager import RecordIndexManager
from mcp_markdown_ragdocs.indexing.record_ports import PreparedRecordDocument
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

    def register_content_source(self, source: ContentSource) -> None:
        del source

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


class _SourceMapStore:
    def __init__(self) -> None:
        self.saved: list[dict[str, list[str]]] = []

    def load(self) -> dict[str, list[str]]:
        return {}

    def save(self, records: Mapping[str, Sequence[str]]) -> None:
        self.saved.append(
            {doc_id: list(storage_keys) for doc_id, storage_keys in records.items()}
        )


class _RecordingWriter:
    def __init__(self, error: str | None = None) -> None:
        self.error = error
        self.calls: list[tuple[str, tuple[str, ...]]] = []

    async def write(
        self,
        prepared: PreparedRecordDocument,
        old_keys: Sequence[str],
    ) -> tuple[str, ...]:
        self.calls.append((prepared.document.id, tuple(old_keys)))
        if self.error is not None:
            raise RuntimeError(self.error)
        return tuple(record.storage_key for record in prepared.records)


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


def test_manager_writes_planned_document_through_ports(record_manager) -> None:
    """
    Route public Markdown indexing through injected planning and writing ports.
    """
    docs_dir = Path(record_manager._config.indexing.documents_path)
    file_path = create_test_document(docs_dir, "guide", "# Guide\n\nManager contract")
    planner = MarkdownDocumentPlanner(
        record_manager._config,
        [docs_dir],
        get_chunker(record_manager._config.chunking),
    )
    source_map = _SourceMapStore()
    writer = _RecordingWriter()
    manager = RecordIndexManager(
        record_manager._config,
        record_manager.kernel,
        record_manager.embedding_provider,
        document_planner=planner,
        document_writer=writer,
        source_map_store=source_map,
    )

    assert manager.index_document(file_path) is True
    manager.persist()

    prepared = planner.plan(file_path)
    assert writer.calls == [(prepared.document.id, ())]
    assert source_map.saved[-1] == {
        prepared.document.id: [record.storage_key for record in prepared.records]
    }
    assert manager.get_state_version() == 1
    assert manager.get_failed_files() == []


def test_manager_preserves_writer_failure_payload(record_manager) -> None:
    """
    Report writer failures without advancing state or saving membership.
    """
    docs_dir = Path(record_manager._config.indexing.documents_path)
    file_path = create_test_document(docs_dir, "guide", "# Guide\n\nManager failure")
    planner = MarkdownDocumentPlanner(
        record_manager._config,
        [docs_dir],
        get_chunker(record_manager._config.chunking),
    )
    source_map = _SourceMapStore()
    writer = _RecordingWriter("writer failed")
    manager = RecordIndexManager(
        record_manager._config,
        record_manager.kernel,
        record_manager.embedding_provider,
        document_planner=planner,
        document_writer=writer,
        source_map_store=source_map,
    )

    assert manager.index_document(file_path) is False
    assert manager.get_failed_files() == [{"path": file_path, "error": "writer failed"}]
    assert manager.get_state_version() == 0
    assert source_map.saved == []
