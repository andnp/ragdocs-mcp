"""Application-owned capabilities for canonical record indexing."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from searchkernel.api import (
    ContentSource,
    GraphEdge,
    GraphNeighbor,
    LocalRecordKernel,
    Record,
    RecordIdentity,
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


class RecordStorage(Protocol):
    """Read and mutate canonical records without exposing a kernel backend."""

    @property
    def db_manager(self) -> object: ...

    def register_content_source(self, source: ContentSource) -> None: ...

    def hydrate_record(self, identity: RecordIdentity | str) -> Record | None: ...

    def hydrate_records(
        self,
        identities: Sequence[RecordIdentity],
    ) -> Mapping[str, Record | None]: ...

    def iter_records(self) -> Iterable[Record]: ...

    def delete(self, storage_keys: Sequence[str]) -> None: ...


class GraphCapability(Protocol):
    """Navigate and mutate record relationships without exposing a kernel."""

    def set_direction(self, direction: str | bool) -> None: ...

    def upsert_edges(self, edges: Sequence[GraphEdge]) -> None: ...

    def delete_edges(self, edges: Sequence[GraphEdge]) -> None: ...

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
    ) -> None:
        self._kernel = kernel
        self._deletion = deletion or LocalRecordDeletion(kernel)
        from mcp_markdown_ragdocs.indexing.local_graph import (
            LocalBidirectionalGraphStore,
        )

        self._graph = LocalBidirectionalGraphStore(
            kernel,
            self.iter_identities,
        )
        self._graph.install()

    @property
    def db_manager(self) -> object:
        return self._kernel.backend.db_manager

    @property
    def keyword_store(self) -> object:
        return self._kernel.keyword_store

    @property
    def vector_store(self) -> object:
        return self._kernel.vector_store

    @property
    def graph(self) -> GraphCapability:
        return self._graph

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

    def iter_records(self) -> Iterable[Record]:
        """Stream records through the local database manager boundary.

        Hydrates in fixed-size batches rather than materialising the whole
        table at once, so callers can process records without holding the
        entire index (tens of thousands of rows) in memory simultaneously.
        """
        identities = self.iter_identities()
        batch_size = self._ITER_RECORDS_BATCH_SIZE
        for start in range(0, len(identities), batch_size):
            chunk = identities[start : start + batch_size]
            hydrated = self.hydrate_records(chunk)
            for identity in chunk:
                record = hydrated.get(identity.storage_key)
                if record is not None:
                    yield record

    def iter_identities(self) -> tuple[RecordIdentity, ...]:
        """Return canonical identities without exposing database rows."""
        connection = self._kernel.backend.db_manager.get_connection()
        rows = connection.execute("SELECT storage_key FROM local_records")
        return tuple(RecordIdentity.from_storage_key(str(row[0])) for row in rows)

    def tune_backend(self) -> None:
        """Apply the application's local SQLite performance settings."""
        try:
            connection = self._kernel.backend.db_manager.get_connection()
            connection.execute("PRAGMA cache_size = -64000")
            connection.execute("PRAGMA mmap_size = 1073741824")
        except Exception:
            return

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
            json.dumps(
                {doc_id: list(keys) for doc_id, keys in records.items()},
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        temporary.replace(self._path)


__all__ = [
    "JsonSourceMapStore",
    "LocalRecordDeletion",
    "LocalRecordStorage",
    "RecordDeletion",
    "RecordStorage",
    "SourceMapStore",
]
