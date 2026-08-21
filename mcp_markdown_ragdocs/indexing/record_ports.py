"""Application-owned capabilities for canonical record indexing."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from typing import Protocol

from searchkernel.api import (
    ContentSource,
    GraphEdge,
    GraphNeighbor,
    LocalRecordKernel,
    KeywordStore,
    Record,
    RecordIdentity,
    SQLiteEmbeddingCache,
    SemanticRecordIngestor,
    Vector,
    VectorStore,
)

from mcp_markdown_ragdocs.models import Document


@dataclass(frozen=True)
class PreparedRecordDocument:
    """A parsed Markdown document plus its canonical chunk records."""

    file_path: str
    document: Document
    records: tuple[Record, ...]


class DocumentPlanner(Protocol):
    """Plan a source file into application document and record values."""

    def plan(self, file_path: str) -> PreparedRecordDocument: ...


class DocumentWriter(Protocol):
    """Write planned document records while removing stale storage keys."""

    async def write(
        self,
        prepared: PreparedRecordDocument,
        old_keys: Sequence[str],
    ) -> tuple[str, ...]: ...


class CommitHistoryPort(Protocol):
    """List commit hashes newer than a timestamp for git record repair."""

    def __call__(
        self,
        git_dir: Path,
        after_timestamp: int | None = None,
    ) -> Iterator[str]: ...


class GDriveIntegrationPort(Protocol):
    """Route Google Drive-sourced records through provider-aware replacement."""

    @property
    def source_kind(self) -> str: ...

    async def replace(self, records: Sequence[Record]) -> None: ...

    def recover(self) -> bool: ...


GDriveIntegrationFactory = Callable[
    [
        Path,
        Mapping[str, ContentSource],
        SemanticRecordIngestor,
        "RecordStorage",
        dict[str, list[str]],
        "SourceMapStore",
    ],
    GDriveIntegrationPort,
]


class SQLiteConnectionProvider(Protocol):
    """Expose the database connection needed by a local storage adapter."""

    def get_connection(self) -> sqlite3.Connection: ...


class RecordIdentityCatalog(Protocol):
    """Enumerate canonical record identities without exposing provider rows."""

    def iter_identities(
        self,
        *,
        source_kind: str | None = None,
        status: str | None = None,
    ) -> Iterator[RecordIdentity]: ...

    def count_distinct_git_commits(self, *, status: str | None = None) -> int: ...


class LocalRecordIdentityCatalog:
    """Temporarily adapt SearchKernel's local identity storage to this port."""

    def __init__(self, database_manager: SQLiteConnectionProvider) -> None:
        self._database_manager = database_manager

    def iter_identities(
        self,
        *,
        source_kind: str | None = None,
        status: str | None = None,
    ) -> Iterator[RecordIdentity]:
        """Stream identities directly from indexed columns, filtered in SQL.

        Bypasses storage-key JSON decoding and full record hydration so
        callers that only need identity fields (not body/metadata) can skip
        both.
        """
        query = "SELECT workspace_id, source_kind, source_id FROM local_records"
        clauses: list[str] = []
        parameters: list[str] = []
        if source_kind is not None:
            clauses.append("source_kind = ?")
            parameters.append(source_kind)
        if status is not None:
            clauses.append("status = ?")
            parameters.append(status)
        if clauses:
            query += " WHERE " + " AND ".join(clauses)

        connection = self._database_manager.get_connection()
        rows = connection.execute(query, parameters)
        for row in rows:
            yield RecordIdentity(
                workspace_id=row[0],
                source_kind=row[1],
                source_id=row[2],
            )

    def count_distinct_git_commits(self, *, status: str | None = None) -> int:
        """Count commit identities with SQL-side aggregation."""
        commit_id = """
            CASE
                WHEN source_id LIKE 'git:%'
                    AND instr(substr(source_id, 5), ':') > 0
                THEN substr(source_id, 1, 3 + instr(substr(source_id, 5), ':'))
                ELSE source_id
            END
        """
        query = f"""
            SELECT COUNT(DISTINCT {commit_id})
            FROM local_records
            WHERE source_kind = ?
        """
        parameters: list[str] = ["git_commit"]
        if status is not None:
            query += " AND status = ?"
            parameters.append(status)

        connection = self._database_manager.get_connection()
        row = connection.execute(query, parameters).fetchone()
        return int(row[0]) if row is not None else 0


class RecordStorage(Protocol):
    """Read and mutate canonical records without exposing a kernel backend."""

    def register_content_source(self, source: ContentSource) -> None: ...

    def hydrate_record(self, identity: RecordIdentity | str) -> Record | None: ...

    def hydrate_records(
        self,
        identities: Sequence[RecordIdentity],
    ) -> Mapping[str, Record | None]: ...

    def iter_records(
        self,
        *,
        source_kind: str | None = None,
        status: str | None = None,
    ) -> Iterable[Record]: ...

    def iter_identities(
        self,
        *,
        source_kind: str | None = None,
        status: str | None = None,
    ) -> Iterator[RecordIdentity]: ...

    def count_distinct_git_commits(self, *, status: str | None = None) -> int: ...

    def run_incremental_vacuum(self, page_limit: int) -> int: ...

    def delete(self, storage_keys: Sequence[str]) -> None: ...


class RecordIndexStorage(RecordStorage, Protocol):
    """Additional local capabilities needed to construct the index writer."""

    @property
    def database_manager(self) -> object: ...

    @property
    def keyword_store(self) -> KeywordStore: ...

    @property
    def vector_store(self) -> VectorStore: ...

    def create_ingestor(
        self,
        embedding_provider: "EmbeddingProvider",
        *,
        cache_path: Path,
        batch_size: int,
    ) -> SemanticRecordIngestor: ...


class EmbeddingProvider(Protocol):
    """Embedding capability required by the local ingestion adapter."""

    model_name: str
    dim: int

    def embed(self, texts: list[str]) -> list[Vector]: ...

    def embed_query(self, text: str) -> Vector: ...


class GraphCapability(Protocol):
    """Navigate and mutate record relationships without exposing a kernel."""

    def set_direction(self, direction: str | bool) -> None: ...

    def upsert_edges(self, edges: Sequence[GraphEdge]) -> None: ...

    def delete_edges(self, edges: Sequence[GraphEdge]) -> None: ...

    def outgoing_edges(
        self,
        identities: Sequence[RecordIdentity],
        edge_type: str,
    ) -> list[GraphEdge]: ...

    def neighbors_many(
        self,
        identities: Sequence[RecordIdentity],
        *,
        depth: int = 1,
        max_neighbors: int | None = None,
    ) -> Mapping[str, Sequence[GraphNeighbor]]: ...

    def neighbors(
        self,
        identity: RecordIdentity,
        *,
        depth: int = 1,
        max_neighbors: int | None = None,
    ) -> Sequence[GraphNeighbor]: ...

    def graph_integrity_errors(self) -> list[str]: ...


class RecordDeletion(Protocol):
    """Delete canonical records from every retrieval surface atomically."""

    def delete(self, storage_keys: Sequence[str]) -> None: ...


class SourceMapStore(Protocol):
    """Persist source-to-record membership with application-owned formatting."""

    def load(self) -> dict[str, list[str]]: ...

    def save(self, records: Mapping[str, Sequence[str]]) -> None: ...


class LocalRecordStorage:
    """Adapt the public local kernel stores to the record manager port."""

    def __init__(
        self,
        kernel: LocalRecordKernel,
        deletion: RecordDeletion | None = None,
        identity_catalog: RecordIdentityCatalog | None = None,
    ) -> None:
        self._kernel = kernel
        self._deletion = deletion or LocalRecordDeletion(kernel)
        self._identity_catalog = identity_catalog or LocalRecordIdentityCatalog(
            kernel.backend.db_manager
        )
        from mcp_markdown_ragdocs.indexing.local_graph import (
            install_bidirectional_graph_store,
        )

        self._graph = install_bidirectional_graph_store(kernel, self.iter_identities)

    @property
    def graph(self) -> GraphCapability:
        return self._graph

    @property
    def database_manager(self) -> object:
        return self._kernel.backend.db_manager

    @property
    def keyword_store(self) -> KeywordStore:
        return self._kernel.keyword_store

    @property
    def vector_store(self) -> VectorStore:
        return self._kernel.vector_store

    def create_ingestor(
        self,
        embedding_provider: EmbeddingProvider,
        *,
        cache_path: Path,
        batch_size: int,
    ) -> SemanticRecordIngestor:
        embedding_cache = SQLiteEmbeddingCache(
            cache_path,
            encoder_namespace=embedding_provider.model_name,
            dimension=embedding_provider.dim,
        )
        return SemanticRecordIngestor(
            embedding_provider=embedding_provider,
            keyword_store=self.keyword_store,
            vector_store=self.vector_store,
            embedding_cache=embedding_cache,
            embedding_batch_size=max(1, batch_size),
        )

    def register_content_source(self, source: ContentSource) -> None:
        self._kernel.kernel.register_content_source(source)

    def hydrate_record(self, identity: RecordIdentity | str) -> Record | None:
        if isinstance(identity, RecordIdentity):
            canonical = identity
        else:
            canonical = RecordIdentity.from_storage_key(identity)
        return self.hydrate_records((canonical,)).get(canonical.storage_key)

    def hydrate_records(
        self,
        identities: Sequence[RecordIdentity],
    ) -> Mapping[str, Record | None]:
        return self._kernel.backend.hydrate_records(identities)

    _ITER_RECORDS_BATCH_SIZE = 500

    def iter_records(
        self,
        *,
        source_kind: str | None = None,
        status: str | None = None,
    ) -> Iterable[Record]:
        """Stream records through the local database manager boundary.

        Hydrates in fixed-size batches rather than materialising the whole
        table at once, so callers can process records without holding the
        entire index (tens of thousands of rows) in memory simultaneously.
        """
        identities = iter(self.iter_identities(source_kind=source_kind, status=status))
        batch_size = self._ITER_RECORDS_BATCH_SIZE
        while chunk := list(islice(identities, batch_size)):
            hydrated = self.hydrate_records(chunk)
            for identity in chunk:
                record = hydrated.get(identity.storage_key)
                if record is not None:
                    yield record

    def iter_identities(
        self,
        *,
        source_kind: str | None = None,
        status: str | None = None,
    ) -> Iterator[RecordIdentity]:
        """Stream canonical identities through the application-owned catalog."""
        yield from self._identity_catalog.iter_identities(
            source_kind=source_kind, status=status
        )

    def count_distinct_git_commits(self, *, status: str | None = None) -> int:
        return self._identity_catalog.count_distinct_git_commits(status=status)

    def tune_backend(self) -> None:
        """Apply the application's local SQLite performance settings."""
        try:
            connection = self._kernel.backend.db_manager.get_connection()
            connection.execute("PRAGMA cache_size = -64000")
            connection.execute("PRAGMA mmap_size = 1073741824")
        except Exception:
            return

    def run_incremental_vacuum(self, page_limit: int) -> int:
        """Reclaim up to page_limit freed pages, returning the pages reclaimed.

        A no-op (returns 0) unless the database has been migrated to
        auto_vacuum=INCREMENTAL; incremental_vacuum otherwise has no freed
        pages to reclaim regardless of the requested limit.
        """
        connection = self._kernel.backend.db_manager.get_connection()
        before = connection.execute("PRAGMA freelist_count").fetchone()[0]
        connection.execute(f"PRAGMA incremental_vacuum({int(page_limit)})")
        after = connection.execute("PRAGMA freelist_count").fetchone()[0]
        return max(0, before - after)

    def delete(self, storage_keys: Sequence[str]) -> None:
        self._deletion.delete(storage_keys)


class LocalRecordDeletion:
    """Adapt the local backend's canonical delete transaction to the port."""

    def __init__(self, kernel: LocalRecordKernel) -> None:
        self._kernel = kernel

    def delete(self, storage_keys: Sequence[str]) -> None:
        self._kernel.backend.delete(list(storage_keys))


class JsonSourceMapStore:
    """Store source-map membership using the legacy JSON representation."""

    def __init__(self, path: Path) -> None:
        self._path = path

    def load(self) -> dict[str, list[str]]:
        try:
            value = json.loads(self._path.read_text(encoding="utf-8"))
        except (FileNotFoundError, OSError, json.JSONDecodeError):
            return {}
        if not isinstance(value, dict):
            return {}
        return {
            str(doc_id): [str(key) for key in keys if isinstance(key, str)]
            for doc_id, keys in value.items()
            if isinstance(keys, list)
        }

    def save(self, records: Mapping[str, Sequence[str]]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self._path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps({doc_id: list(keys) for doc_id, keys in records.items()}),
            encoding="utf-8",
        )
        temporary.replace(self._path)


__all__ = [
    "JsonSourceMapStore",
    "CommitHistoryPort",
    "EmbeddingProvider",
    "GDriveIntegrationFactory",
    "GDriveIntegrationPort",
    "RecordIndexStorage",
    "LocalRecordDeletion",
    "LocalRecordIdentityCatalog",
    "LocalRecordStorage",
    "RecordDeletion",
    "RecordIdentityCatalog",
    "RecordStorage",
    "SQLiteConnectionProvider",
    "SourceMapStore",
]
