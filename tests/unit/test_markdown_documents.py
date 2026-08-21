"""Contract tests for Markdown planning and canonical document writing."""

import asyncio
from collections.abc import Iterable, Iterator, Mapping, Sequence
from datetime import UTC, datetime
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
from mcp_markdown_ragdocs.models import Document
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
        self.calls: list[tuple[Record, ...]] = []

    async def index_records(
        self,
        records: Sequence[Record],
        *,
        checkpoint: Cursor | None = None,
        failure_mode: IngestionFailureMode = "strict",
    ) -> IngestionReceipt:
        del checkpoint, failure_mode
        self.calls.append(tuple(records))
        self.events.append("index")
        return self.receipt


class _Storage:
    def __init__(
        self,
        events: list[str],
        existing_records: Mapping[str, Record] | None = None,
    ) -> None:
        self.events = events
        self.deleted: list[tuple[str, ...]] = []
        self._existing_records = dict(existing_records or {})

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
        return {
            identity.storage_key: self._existing_records.get(identity.storage_key)
            for identity in identities
        }

    def iter_records(
        self, *, source_kind: str | None = None, status: str | None = None
    ) -> Iterable[Record]:
        return ()

    def iter_identities(
        self, *, source_kind: str | None = None, status: str | None = None
    ) -> Iterator[RecordIdentity]:
        return iter(())

    def count_distinct_git_commits(self, *, status: str | None = None) -> int:
        del status
        return 0

    def run_incremental_vacuum(self, page_limit: int) -> int:
        del page_limit
        return 0

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


def _chunk_record(chunk_id: str, body: str) -> Record:
    when = datetime(2024, 1, 1, tzinfo=UTC)
    return Record(
        source_kind="note",
        source_id=chunk_id,
        title=f"Guide / {chunk_id}",
        body=body,
        created_at=when,
        updated_at=when,
        metadata={"chunk_id": chunk_id, "doc_id": "guide"},
        workspace_id="default",
        indexed_text=body,
    )


def _prepared_document(*records: Record) -> PreparedRecordDocument:
    when = datetime(2024, 1, 1, tzinfo=UTC)
    return PreparedRecordDocument(
        "guide.md",
        Document(
            id="guide",
            content="irrelevant",
            metadata={},
            links=[],
            tags=[],
            file_path="guide.md",
            modified_time=when,
        ),
        tuple(records),
    )


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


def test_writer_reindexes_only_the_forced_first_chunk_when_others_are_unchanged() -> None:
    """
    Skip re-embedding unchanged chunks, but always refresh keys[0].
    """
    first = _chunk_record("guide::0", "First chunk body")
    second = _chunk_record("guide::1", "Second chunk body")
    prepared = _prepared_document(first, second)
    events: list[str] = []
    ingestor = _Ingestor(events, _success_receipt())
    storage = _Storage(
        events,
        existing_records={first.storage_key: first, second.storage_key: second},
    )
    writer = SemanticDocumentWriter(ingestor, storage)

    new_keys = asyncio.run(writer.write(prepared, (first.storage_key, second.storage_key)))

    assert new_keys == (first.storage_key, second.storage_key)
    assert ingestor.calls == [(first,)]
    assert storage.deleted == []


def test_writer_reindexes_a_genuinely_changed_chunk_plus_the_forced_first_chunk() -> None:
    """
    Reindex a chunk whose content changed, alongside the always-forced first chunk.
    """
    first = _chunk_record("guide::0", "First chunk body")
    old_second = _chunk_record("guide::1", "Second chunk body")
    new_second = _chunk_record("guide::1", "Second chunk body, edited")
    prepared = _prepared_document(first, new_second)
    events: list[str] = []
    ingestor = _Ingestor(events, _success_receipt())
    storage = _Storage(
        events,
        existing_records={first.storage_key: first, old_second.storage_key: old_second},
    )
    writer = SemanticDocumentWriter(ingestor, storage)

    new_keys = asyncio.run(
        writer.write(prepared, (first.storage_key, old_second.storage_key))
    )

    assert new_keys == (first.storage_key, new_second.storage_key)
    assert ingestor.calls == [(first, new_second)]
    assert storage.deleted == []


def test_writer_deletes_stale_keys_even_when_no_chunk_content_changed() -> None:
    """
    Delete a stale key for a removed chunk even though remaining chunks are unchanged.

    Stale-key deletion must stay unconditional on old_keys/new_keys, independent
    of whether any chunk content changed and thus whether index_records ran with
    more than the forced first chunk.
    """
    first = _chunk_record("guide::0", "First chunk body")
    removed = _chunk_record("guide::removed", "Chunk that no longer exists")
    prepared = _prepared_document(first)
    events: list[str] = []
    ingestor = _Ingestor(events, _success_receipt())
    storage = _Storage(events, existing_records={first.storage_key: first})
    writer = SemanticDocumentWriter(ingestor, storage)

    new_keys = asyncio.run(
        writer.write(prepared, (first.storage_key, removed.storage_key))
    )

    assert new_keys == (first.storage_key,)
    assert ingestor.calls == [(first,)]
    assert storage.deleted == [(removed.storage_key,)]
