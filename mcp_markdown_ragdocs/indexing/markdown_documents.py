"""Markdown planning and canonical record writing adapters."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

from searchkernel.api import (
    Record,
    RecordChunker,
    RecordIdentity,
    RecordIngestor,
    RecordStatus,
)

from mcp_markdown_ragdocs.config import Config, resolve_project_id_for_path
from mcp_markdown_ragdocs.indexing.record_ports import (
    DocumentPlanner,
    DocumentWriter,
    PreparedRecordDocument,
    RecordStorage,
)
from mcp_markdown_ragdocs.models import Document
from mcp_markdown_ragdocs.parsers.dispatcher import dispatch_parser

_FILE_MTIME_METADATA_KEY = "_file_mtime_ns"
_FILE_SIZE_METADATA_KEY = "_file_size"
_VOLATILE_METADATA_KEYS = frozenset({_FILE_MTIME_METADATA_KEY, _FILE_SIZE_METADATA_KEY})


def _stable_metadata(metadata: Mapping[str, object]) -> dict[str, object]:
    """Drop file-stat metadata that changes on every save regardless of content."""
    return {key: value for key, value in metadata.items() if key not in _VOLATILE_METADATA_KEYS}


def _record_unchanged(candidate: Record, existing: Record | None) -> bool:
    """Compare a planned chunk record against its previously stored version."""
    if existing is None:
        return False
    return (
        candidate.body == existing.body
        and candidate.indexed_text == existing.indexed_text
        and candidate.title == existing.title
        and _stable_metadata(candidate.metadata) == _stable_metadata(existing.metadata)
    )


class MarkdownDocumentPlanner:
    """Build stable Markdown document identities and chunk records."""

    def __init__(
        self,
        config: Config,
        documents_roots: Sequence[Path],
        chunker: RecordChunker,
    ) -> None:
        self._config = config
        self._chunker = chunker
        self._documents_roots = [root.resolve() for root in documents_roots]

    def _doc_id_for_path(self, file_path: str) -> str:
        from searchkernel.api import compute_doc_id, compute_doc_id_multi_root

        path = Path(file_path).resolve()
        if len(self._documents_roots) == 1:
            return compute_doc_id(path, self._documents_roots[0])
        return compute_doc_id_multi_root(path, self._documents_roots)

    def _document_record(self, document: Document) -> Record:
        return Record(
            source_kind="note",
            source_id=document.id,
            title=document.id,
            body=document.content,
            created_at=document.modified_time,
            updated_at=document.modified_time,
            metadata={
                "links": document.links,
                "tags": document.tags,
                "file_path": document.file_path,
                "project_id": document.project_id,
                **document.metadata,
            },
            uri=f"file://{document.file_path}",
            status=RecordStatus.ACTIVE,
        )

    def _chunk_records(self, document: Document) -> tuple[Record, ...]:
        chunks = self._chunker.chunk_record(self._document_record(document))
        records: list[Record] = []
        for chunk in chunks:
            metadata = {
                **document.metadata,
                "chunk_id": chunk.chunk_id,
                "doc_id": document.id,
                "chunk_index": chunk.chunk_index,
                "file_path": document.file_path,
                "project_id": document.project_id,
                "links": document.links,
                "tags": document.tags,
                **chunk.metadata,
            }
            metadata.setdefault(
                "_chunk_parent_storage_key",
                RecordIdentity(
                    document.project_id,
                    "note",
                    chunk.chunk_id,
                ).storage_key,
            )
            records.append(
                Record(
                    source_kind="note",
                    source_id=chunk.chunk_id,
                    title=str(chunk.metadata.get("header_path") or document.id),
                    body=chunk.content,
                    created_at=document.modified_time,
                    updated_at=document.modified_time,
                    metadata=metadata,
                    uri=f"file://{document.file_path}",
                    status=RecordStatus.ACTIVE,
                    workspace_id=document.project_id,
                    indexed_text=chunk.content,
                )
            )
        return tuple(records)

    def plan(self, file_path: str) -> PreparedRecordDocument:
        parser = dispatch_parser(file_path)
        document = parser.parse(file_path)
        document.id = self._doc_id_for_path(file_path)
        document.project_id = resolve_project_id_for_path(Path(file_path), self._config)
        try:
            file_stat = Path(file_path).resolve().stat()
        except OSError:
            file_stat = None
        if file_stat is not None:
            document.metadata = {
                **document.metadata,
                _FILE_MTIME_METADATA_KEY: file_stat.st_mtime_ns,
                _FILE_SIZE_METADATA_KEY: file_stat.st_size,
            }
        return PreparedRecordDocument(
            file_path,
            document,
            self._chunk_records(document),
        )


class SemanticDocumentWriter:
    """Persist planned records and remove stale keys in canonical order."""

    def __init__(self, ingestor: RecordIngestor, storage: RecordStorage) -> None:
        self._ingestor = ingestor
        self._storage = storage

    async def write(
        self,
        prepared: PreparedRecordDocument,
        old_keys: Sequence[str],
    ) -> tuple[str, ...]:
        records = prepared.records
        new_keys = tuple(record.storage_key for record in records)
        reindex_records = self._select_reindex_records(records)
        if reindex_records:
            receipt = await self._ingestor.index_records(reindex_records)
            if receipt.failed:
                errors = "; ".join(
                    item.error or "unknown error" for item in receipt.failures
                )
                raise RuntimeError(errors)
        stale_keys = sorted(set(old_keys) - set(new_keys))
        if stale_keys:
            self._storage.delete(stale_keys)
        return new_keys

    def _select_reindex_records(
        self,
        records: Sequence[Record],
    ) -> tuple[Record, ...]:
        """Skip re-embedding chunks unchanged since the last save.

        keys[0] always stays reindexed: record_manager freshness checks only
        inspect the first chunk's stored file-stat metadata, so it must never
        go stale even when its own content is unchanged.
        """
        if not records:
            return ()
        existing = self._storage.hydrate_records(
            tuple(
                RecordIdentity.from_storage_key(record.storage_key)
                for record in records
            )
        )
        return tuple(
            record
            for index, record in enumerate(records)
            if index == 0
            or not _record_unchanged(record, existing.get(record.storage_key))
        )


__all__ = [
    "DocumentPlanner",
    "DocumentWriter",
    "MarkdownDocumentPlanner",
    "PreparedRecordDocument",
    "SemanticDocumentWriter",
]
