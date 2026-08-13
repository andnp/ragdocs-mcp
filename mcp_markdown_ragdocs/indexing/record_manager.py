"""Canonical record indexing for Markdown sources.

The application owns file discovery and parsing.  Searchkernel owns durable
record storage, embedding, and retrieval.  This module is the small seam that
connects those two responsibilities.
"""

from __future__ import annotations

import asyncio
import contextvars
import hashlib
import json
import logging
import os
import re
from collections.abc import Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, cast
from urllib.parse import unquote, urlparse

from searchkernel.api import (
    ContentSource,
    GraphEdge,
    GraphNeighbor,
    LocalRecordKernel,
    OllamaEmbeddingProvider,
    Record,
    RecordIdentity,
    RecordStatus,
    SemanticRecordIngestor,
    SQLiteEmbeddingCache,
    Vector,
    compute_doc_id,
    compute_doc_id_multi_root,
    get_chunker,
)

from mcp_markdown_ragdocs.config import Config, resolve_project_id_for_path
from mcp_markdown_ragdocs.git.repository import iter_commit_hashes_after_timestamp
from mcp_markdown_ragdocs.gdrive.replacement import (
    GDriveReplacementJournal,
    REPLACEMENT_JOURNAL_FILENAME,
    canonical_gdrive_source_id,
    canonical_gdrive_source_key,
    group_gdrive_records,
    is_gdrive_tombstone,
)
from mcp_markdown_ragdocs.gdrive.records import SOURCE_KIND as GDRIVE_SOURCE_KIND
from mcp_markdown_ragdocs.gdrive.state import (
    GDriveScopeIdentity,
    GDriveStateRepository,
)
from mcp_markdown_ragdocs.models import Document
from mcp_markdown_ragdocs.parsers.dispatcher import dispatch_parser

logger = logging.getLogger(__name__)
_GRAPH_IDENTITY_BATCH_SIZE = 100
_FILE_MTIME_METADATA_KEY = "_file_mtime_ns"
_FILE_SIZE_METADATA_KEY = "_file_size"


class _BidirectionalGraphStore:
    def __init__(self, graph_store: Any, identities: Any) -> None:
        self._graph_store = graph_store
        self._identities = identities
        self._direction = contextvars.ContextVar(
            "graph_direction",
            default="outgoing",
        )

    def set_direction(self, direction: str | bool) -> None:
        normalized = (
            "incoming"
            if direction is True
            else "outgoing"
            if direction is False
            else direction
        )
        self._direction.set(normalized)

    def neighbors_many(
        self,
        identities: Sequence[RecordIdentity],
        *,
        depth: int,
        max_neighbors: int | None = None,
    ) -> dict[str, Sequence[GraphNeighbor]]:
        direction = self._direction.get()
        if direction == "incoming":
            return self.incoming_neighbors_many(
                identities,
                depth=depth,
                max_neighbors=max_neighbors,
            )
        if direction == "both":
            outgoing = self._outgoing_neighbors_many(
                identities,
                depth=depth,
                max_neighbors=max_neighbors,
            )
            incoming = self.incoming_neighbors_many(
                identities,
                depth=depth,
                max_neighbors=max_neighbors,
            )
            return {
                identity.storage_key: _merge_graph_neighbors(
                    outgoing.get(identity.storage_key, ()),
                    incoming.get(identity.storage_key, ()),
                    max_neighbors,
                )
                for identity in identities
            }
        return self._outgoing_neighbors_many(
            identities,
            depth=depth,
            max_neighbors=max_neighbors,
        )

    def _outgoing_neighbors_many(
        self,
        identities: Sequence[RecordIdentity],
        *,
        depth: int,
        max_neighbors: int | None = None,
    ) -> dict[str, Sequence[GraphNeighbor]]:
        return cast(
            dict[str, Sequence[GraphNeighbor]],
            self._graph_store.neighbors_many(
                identities,
                depth=depth,
                max_neighbors=max_neighbors,
            ),
        )

    def neighbors(
        self,
        identity: RecordIdentity,
        *,
        depth: int,
        max_neighbors: int | None = None,
    ) -> Sequence[GraphNeighbor]:
        return self.neighbors_many(
            [identity],
            depth=depth,
            max_neighbors=max_neighbors,
        )[identity.storage_key]

    def incoming_neighbors(
        self,
        identity: RecordIdentity,
        *,
        depth: int,
        max_neighbors: int | None = None,
    ) -> Sequence[GraphNeighbor]:
        return self.incoming_neighbors_many(
            [identity],
            depth=depth,
            max_neighbors=max_neighbors,
        )[identity.storage_key]

    def incoming_neighbors_many(
        self,
        identities: Sequence[RecordIdentity],
        *,
        depth: int,
        max_neighbors: int | None = None,
    ) -> dict[str, Sequence[GraphNeighbor]]:
        native_loader = getattr(self._graph_store, "incoming_neighbors_many", None)
        if callable(native_loader):
            return cast(
                dict[str, Sequence[GraphNeighbor]],
                native_loader(
                    identities,
                    depth=depth,
                    max_neighbors=max_neighbors,
                ),
            )
        requested = {identity.storage_key for identity in identities}
        incoming: dict[str, list[GraphNeighbor]] = {
            identity.storage_key: [] for identity in identities
        }
        outgoing: dict[str, Sequence[GraphNeighbor]] = {}
        all_identities = self._identities()
        for start in range(0, len(all_identities), _GRAPH_IDENTITY_BATCH_SIZE):
            outgoing.update(
                self._graph_store.neighbors_many(
                    all_identities[start : start + _GRAPH_IDENTITY_BATCH_SIZE],
                    depth=depth,
                    max_neighbors=None,
                )
            )
        for source_key, neighbors in outgoing.items():
            source = RecordIdentity.from_storage_key(source_key)
            for neighbor in neighbors:
                if neighbor.identity.storage_key not in requested:
                    continue
                incoming[neighbor.identity.storage_key].append(
                    GraphNeighbor(source, neighbor.edge_type, neighbor.weight)
                )
        for key, neighbors in incoming.items():
            neighbors.sort(key=lambda item: (-item.weight, item.identity.storage_key))
            if max_neighbors is not None:
                incoming[key] = neighbors[:max_neighbors]
        return cast(dict[str, Sequence[GraphNeighbor]], incoming)


def _merge_graph_neighbors(first, second, max_neighbors: int | None):
    merged = {
        neighbor.identity.storage_key: neighbor
        for neighbor in (*first, *second)
    }
    neighbors = sorted(
        merged.values(),
        key=lambda item: (-item.weight, item.identity.storage_key),
    )
    return neighbors if max_neighbors is None else neighbors[:max_neighbors]


def install_bidirectional_graph_store(
    kernel: LocalRecordKernel,
    identities: Any,
) -> None:
    kernel.pipeline._graph_store = _BidirectionalGraphStore(  # type: ignore[attr-defined]
        kernel.graph_store,
        identities,
    )


def _git_commit_id(source_id: str) -> str:
    parts = source_id.split(":")
    return ":".join(parts[:2]) if len(parts) >= 2 and parts[0] == "git" else source_id


class _FakeEmbeddingProvider:
    """Small provider adapter used by the app's deterministic test mode."""

    model_name = "__deterministic_fake__"
    dim = 384

    def __init__(self) -> None:
        self._model = _DeterministicFakeEmbeddingModel(self.dim)

    def embed(self, texts: list[str]) -> list[Vector]:
        return [self._model.vector_for_text(text) for text in texts]


class _DeterministicFakeEmbeddingModel:
    def __init__(self, dimension: int) -> None:
        self.dimension = dimension

    def vector_for_text(self, text: str) -> Vector:
        vector = [0.0] * self.dimension
        tokens = re.findall(r"\b[a-zA-Z0-9_]+\b", text.lower()) or ["__empty__"]
        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            primary = int.from_bytes(digest[:8], "big") % self.dimension
            secondary = int.from_bytes(digest[8:16], "big") % self.dimension
            sign = 1.0 if digest[16] % 2 == 0 else -1.0
            vector[primary] += 1.0
            vector[secondary] += 0.5 * sign
        norm = sum(value * value for value in vector) ** 0.5
        if norm > 0:
            vector = [value / norm for value in vector]
        return vector


def build_embedding_provider(config: Config, model_name: str):
    """Build the one embedding provider shared by indexing and querying."""

    if os.getenv("MCP_RAGDOCS_TEST_FAKE_EMBEDDINGS") == "1":
        return _FakeEmbeddingProvider()

    provider_name = config.embedding.provider.lower()
    if provider_name != "ollama":
        raise ValueError(
            "canonical indexing requires embedding.provider = 'ollama'; "
            f"got {config.embedding.provider!r}"
        )

    return OllamaEmbeddingProvider(
        model_name,
        base_url=config.embedding.base_url,
        dim=config.embedding.dimension,
        timeout=config.embedding.timeout_seconds,
        auto_pull=config.embedding.auto_pull,
        pull_timeout=config.embedding.pull_timeout_seconds,
    )


@dataclass(frozen=True)
class PreparedRecordDocument:
    file_path: str
    document: Document
    records: tuple[Record, ...]


class RecordIndexManager:
    """Index Markdown and source records through searchkernel's local kernel."""

    def __init__(
        self,
        config: Config,
        kernel: LocalRecordKernel,
        embedding_provider: Any,
        *,
        documents_roots: list[Path] | None = None,
        content_sources: Iterable[ContentSource] = (),
    ) -> None:
        self._config = config
        self.kernel = kernel
        self.embedding_provider = embedding_provider
        self._chunker = get_chunker(config.chunking)
        self._documents_roots = [
            root.resolve()
            for root in (documents_roots or [Path(config.indexing.documents_path)])
        ]
        self._failed_files: list[dict[str, str]] = []
        self._state_version = 0
        self._ready = True
        self._source_map_path = Path(config.indexing.index_path) / "record-sources.json"
        self._source_records: dict[str, list[str]] = self._load_source_map()
        self._content_sources: dict[str, ContentSource] = {}
        for source in content_sources:
            self.register_content_source(source)
        self._gdrive_state_repository = (
            GDriveStateRepository(Path(config.indexing.index_path) / "gdrive-state.db")
            if GDRIVE_SOURCE_KIND in self._content_sources
            else None
        )
        self._gdrive_replacement_journal = GDriveReplacementJournal(
            Path(config.indexing.index_path) / REPLACEMENT_JOURNAL_FILENAME,
            self._gdrive_state_repository,
        )
        self._embedding_cache = SQLiteEmbeddingCache(
            Path(config.indexing.index_path) / "embedding-cache.db",
            encoder_namespace=embedding_provider.model_name,
            dimension=embedding_provider.dim,
        )
        self.ingestor = SemanticRecordIngestor(
            embedding_provider=embedding_provider,
            keyword_store=kernel.keyword_store,
            vector_store=kernel.vector_store,
            embedding_cache=self._embedding_cache,
            embedding_batch_size=max(1, config.embedding.batch_size),
        )
        self._recover_gdrive_replacements()

    @property
    def index_path(self) -> Path:
        return Path(self._config.indexing.index_path)

    @property
    def vector(self):
        """Compatibility view for callers that only need semantic operations."""
        return self.kernel.vector_store

    @property
    def keyword(self):
        return self.kernel.keyword_store

    @property
    def graph(self):
        return self.kernel.graph_store

    def is_ready(self) -> bool:
        return self._ready

    def get_state_version(self) -> int:
        return self._state_version

    def get_document_count(self) -> int:
        return len(self._source_records)

    def get_failed_files(self) -> list[dict[str, str]]:
        return list(self._failed_files)

    @property
    def content_sources(self) -> tuple[ContentSource, ...]:
        return tuple(self._content_sources.values())

    def get_content_source(self, source_kind: str) -> ContentSource | None:
        return self._content_sources.get(source_kind)

    def register_content_source(self, source: ContentSource) -> None:
        self.kernel.kernel.register_content_source(source)
        self._content_sources[source.source_kind] = source

    def count_records(self, source_kind: str | None = None) -> int:
        if source_kind is None:
            return sum(len(keys) for keys in self._source_records.values())
        if source_kind == "git_commit":
            return len(
                {
                    _git_commit_id(RecordIdentity.from_storage_key(key).source_id)
                    for keys in self._source_records.values()
                    for key in keys
                    if RecordIdentity.from_storage_key(key).source_kind == source_kind
                }
            )
        count = 0
        for keys in self._source_records.values():
            count += sum(
                RecordIdentity.from_storage_key(key).source_kind == source_kind
                for key in keys
            )
        return count

    def describe_documents(self) -> list[dict[str, object]]:
        descriptions: list[dict[str, object]] = []
        for doc_id, keys in sorted(self._source_records.items()):
            records = [self.kernel.backend.hydrate_record(key) for key in keys]
            record = next((item for item in records if item is not None), None)
            if record is None:
                continue
            descriptions.append(
                {
                    "doc_id": doc_id,
                    "file_path": record.metadata.get("file_path"),
                    "chunk_count": len(keys),
                    "source_kind": record.source_kind,
                }
            )
        return descriptions

    def index_documents(
        self,
        file_paths: list[str],
        force: bool = False,
        persist: bool = False,
    ) -> None:
        candidate_paths = (
            list(file_paths)
            if force
            else [path for path in file_paths if not self._document_is_current(path)]
        )
        indexed = False
        for file_path in candidate_paths:
            indexed = self.index_document(
                file_path,
                update_graph=False,
                force=force,
            ) or indexed
        if indexed:
            self._rebuild_graph()
        if persist:
            self.persist()

    def remove_documents(self, doc_ids: list[str], persist: bool = False) -> None:
        for doc_id in doc_ids:
            self.remove_document(doc_id)
        if persist:
            self.persist()

    def _doc_id_for_path(self, file_path: str) -> str:
        path = Path(file_path).resolve()
        if len(self._documents_roots) == 1:
            return compute_doc_id(path, self._documents_roots[0])
        return compute_doc_id_multi_root(path, self._documents_roots)

    def _document_is_current(self, file_path: str) -> bool:
        try:
            file_stat = Path(file_path).resolve().stat()
        except OSError:
            return False

        keys = self._source_records.get(self._doc_id_for_path(file_path), [])
        if not keys:
            return False
        record = self.kernel.backend.hydrate_record(keys[0])
        if record is None:
            return False
        return (
            record.metadata.get(_FILE_MTIME_METADATA_KEY) == file_stat.st_mtime_ns
            and record.metadata.get(_FILE_SIZE_METADATA_KEY) == file_stat.st_size
        )

    def _load_source_map(self) -> dict[str, list[str]]:
        try:
            value = json.loads(self._source_map_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, OSError, json.JSONDecodeError):
            return {}
        if not isinstance(value, dict):
            return {}
        return {
            str(doc_id): [str(key) for key in keys if isinstance(key, str)]
            for doc_id, keys in value.items()
            if isinstance(keys, list)
        }

    def _save_source_map(self) -> None:
        self._source_map_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self._source_map_path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(self._source_records, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        temporary.replace(self._source_map_path)

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

    def prepare_document(self, file_path: str) -> PreparedRecordDocument:
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
        records = self._chunk_records(document)
        return PreparedRecordDocument(file_path, document, records)

    async def _index_prepared(
        self,
        prepared: PreparedRecordDocument,
        *,
        update_graph: bool = True,
    ) -> None:
        old_keys = self._source_records.get(prepared.document.id, [])
        receipt = await self.ingestor.index_records(prepared.records)
        if receipt.failed:
            errors = "; ".join(item.error or "unknown error" for item in receipt.failures)
            raise RuntimeError(errors)
        new_keys = [record.storage_key for record in prepared.records]
        stale_keys = sorted(set(old_keys) - set(new_keys))
        if stale_keys:
            self.kernel.backend.delete(stale_keys)
        self._source_records[prepared.document.id] = new_keys
        if update_graph:
            self._rebuild_graph()
        self._state_version += 1

    def index_document(
        self,
        file_path: str,
        force: bool = False,
        *,
        update_graph: bool = True,
    ) -> bool:
        try:
            if not force and self._document_is_current(file_path):
                logger.debug("Skipping unchanged document: %s", file_path)
                return True
            prepared = self.prepare_document(file_path)
            _run_async(
                self._index_prepared(
                    prepared,
                    update_graph=update_graph,
                )
            )
            self._save_source_map()
            return True
        except Exception as error:
            self._failed_files.append(
                {"path": file_path, "error": str(error)}
            )
            logger.exception("Failed to index %s", file_path)
            return False

    def index_record(self, record: Record) -> bool:
        if record.source_kind == GDRIVE_SOURCE_KIND:
            return self._index_gdrive_records((record,))
        try:
            receipt = _run_async(self.ingestor.index_records([record]))
            if receipt.failed:
                return False
            self._source_records.setdefault(record.source_id, []).append(
                record.storage_key
            )
            self._save_source_map()
            self._state_version += 1
            return True
        except Exception:
            logger.exception("Failed to index record %s", record.storage_key)
            return False

    def index_records(self, records: Sequence[Record]) -> bool:
        if not records:
            try:
                receipt = _run_async(self.ingestor.index_records(records))
                if receipt.failed:
                    return False
                self._save_source_map()
                self._state_version += 1
                return True
            except Exception:
                logger.exception("Failed to index record batch")
                return False
        gdrive_records = tuple(
            record for record in records if record.source_kind == GDRIVE_SOURCE_KIND
        )
        generic_records = tuple(
            record for record in records if record.source_kind != GDRIVE_SOURCE_KIND
        )
        if generic_records:
            try:
                receipt = _run_async(self.ingestor.index_records(generic_records))
                if receipt.failed:
                    return False
                for record in generic_records:
                    self._source_records.setdefault(record.source_id, []).append(
                        record.storage_key
                    )
                self._save_source_map()
                self._state_version += 1
            except Exception:
                logger.exception("Failed to index record batch")
                return False
        if gdrive_records:
            return self._index_gdrive_records(gdrive_records)
        return bool(generic_records)

    def _index_gdrive_records(self, records: Sequence[Record]) -> bool:
        try:
            for source_key, grouped in group_gdrive_records(records).items():
                self._replace_gdrive_source(source_key, grouped)
            self._rebuild_graph()
            self._state_version += 1
            return True
        except Exception as error:
            logger.exception("Failed to index Google Drive record batch")
            self._failed_files.append({"path": "gdrive", "error": str(error)})
            return False

    def _replace_gdrive_source(
        self,
        source_key: str,
        records: Sequence[Record],
    ) -> None:
        representative = records[0]
        old_keys = self._gdrive_record_keys(source_key)
        tombstones = tuple(record for record in records if is_gdrive_tombstone(record))
        if tombstones:
            self._replace_gdrive_tombstone(source_key, old_keys, records, tombstones)
            return

        records = self._with_gdrive_memberships(records)
        new_keys = tuple(dict.fromkeys(record.storage_key for record in records))
        identities = self._gdrive_scope_identities(representative, records)
        entry = self._gdrive_replacement_journal.prepare(
            source_key,
            old_keys,
            new_keys,
            identities,
        )
        receipt = _run_async(self.ingestor.index_records(records))
        if receipt.failed:
            raise RuntimeError(
                "; ".join(item.error or "unknown error" for item in receipt.failures)
            )
        self._gdrive_replacement_journal.mark_indexed(entry, identities)
        self._delete_gdrive_stale_keys(source_key, old_keys, new_keys)
        self._source_records[source_key] = list(new_keys)
        self._save_source_map()
        self._record_gdrive_memberships(records)
        self._gdrive_replacement_journal.complete(source_key, identities)

    def _replace_gdrive_tombstone(
        self,
        source_key: str,
        old_keys: Sequence[str],
        records: Sequence[Record],
        tombstones: Sequence[Record],
    ) -> None:
        existing = tuple(
            record
            for key in old_keys
            if (record := self.kernel.backend.hydrate_record(key)) is not None
            and not is_gdrive_tombstone(record)
        )
        known_scopes = set(self._gdrive_scopes_from_records(existing))
        removed_scopes = set(self._gdrive_scopes_from_records(tombstones))
        if self._gdrive_state_repository is not None:
            workspace_id = next(
                (record.workspace_id for record in (*records, *existing) if record.workspace_id),
                None,
            )
            source_id = canonical_gdrive_source_id(records[0])
            if workspace_id:
                durable_scopes = self._gdrive_state_repository.memberships_for_source(
                    GDRIVE_SOURCE_KIND,
                    workspace_id,
                    source_id,
                )
                known_scopes.update(durable_scopes)
        if not removed_scopes:
            removed_scopes = set(known_scopes)
        remaining_scopes = tuple(sorted(known_scopes - removed_scopes))
        replacement_records = (
            tuple(self._record_with_scopes(record, remaining_scopes) for record in existing)
            if remaining_scopes
            else ()
        )
        new_keys = tuple(dict.fromkeys(record.storage_key for record in replacement_records))
        identities = self._gdrive_scope_identities(records[0], (*records, *existing))
        entry = self._gdrive_replacement_journal.prepare(
            source_key,
            old_keys,
            new_keys,
            identities,
        )
        if replacement_records:
            receipt = _run_async(self.ingestor.index_records(replacement_records))
            if receipt.failed:
                raise RuntimeError(
                    "; ".join(item.error or "unknown error" for item in receipt.failures)
                )
        self._gdrive_replacement_journal.mark_indexed(entry, identities)
        self._delete_gdrive_stale_keys(source_key, old_keys, new_keys)
        if new_keys:
            self._source_records[source_key] = list(new_keys)
        else:
            self._source_records.pop(source_key, None)
        self._save_source_map()
        self._remove_gdrive_memberships(tombstones, tuple(sorted(removed_scopes)))
        self._gdrive_replacement_journal.complete(source_key, identities)

    def _gdrive_record_keys(self, source_key: str) -> tuple[str, ...]:
        candidates: set[str] = set()
        for keys in self._source_records.values():
            candidates.update(keys)
        for row in self.kernel.backend._record_rows():
            candidates.add(str(row["storage_key"]))
        matching: list[str] = []
        for key in sorted(candidates):
            record = self.kernel.backend.hydrate_record(key)
            if record is None or record.source_kind != GDRIVE_SOURCE_KIND:
                continue
            if canonical_gdrive_source_key(record) == source_key:
                matching.append(key)
        return tuple(matching)

    def _delete_gdrive_stale_keys(
        self,
        source_key: str,
        old_keys: Sequence[str],
        new_keys: Sequence[str],
    ) -> None:
        stale_keys = sorted(set(old_keys) - set(new_keys))
        if stale_keys:
            self.kernel.backend.delete(stale_keys)
        new_key_set = set(new_keys)
        for doc_id, keys in tuple(self._source_records.items()):
            retained = [
                key
                for key in dict.fromkeys(keys)
                if key not in stale_keys
                and not (key in new_key_set and doc_id != source_key)
                and not (doc_id == source_key and key not in new_key_set)
            ]
            if retained:
                self._source_records[doc_id] = retained
            else:
                self._source_records.pop(doc_id, None)

    def _with_gdrive_memberships(self, records: Sequence[Record]) -> tuple[Record, ...]:
        scopes = set(self._gdrive_scopes_from_records(records))
        source_id = canonical_gdrive_source_id(records[0])
        workspace_id = records[0].workspace_id
        if self._gdrive_state_repository is not None and workspace_id:
            scopes.update(
                self._gdrive_state_repository.memberships_for_source(
                    GDRIVE_SOURCE_KIND,
                    workspace_id,
                    source_id,
                )
            )
        return tuple(self._record_with_scopes(record, tuple(sorted(scopes))) for record in records)

    def _record_with_scopes(self, record: Record, scopes: Sequence[str]) -> Record:
        if not scopes:
            return record
        metadata = {**record.metadata, "scope_memberships": list(scopes)}
        return replace(record, metadata=metadata)

    def _gdrive_scopes_from_records(self, records: Sequence[Record]) -> tuple[str, ...]:
        scopes: set[str] = set()
        for record in records:
            raw_scopes = record.metadata.get("scope_memberships")
            if isinstance(raw_scopes, (list, tuple, set)):
                scopes.update(scope for scope in raw_scopes if isinstance(scope, str) and scope)
        return tuple(sorted(scopes))

    def _gdrive_scope_identities(
        self,
        representative: Record,
        records: Sequence[Record],
    ) -> tuple[GDriveScopeIdentity, ...]:
        if not representative.workspace_id:
            return ()
        return tuple(
            GDriveScopeIdentity(GDRIVE_SOURCE_KIND, representative.workspace_id, scope)
            for scope in self._gdrive_scopes_from_records(records)
        )

    def _record_gdrive_memberships(self, records: Sequence[Record]) -> None:
        repository = self._gdrive_state_repository
        if repository is None:
            return
        for record in records:
            if not record.workspace_id:
                continue
            source_id = canonical_gdrive_source_id(record)
            for scope in self._gdrive_scopes_from_records((record,)):
                repository.add_membership(
                    GDriveScopeIdentity(GDRIVE_SOURCE_KIND, record.workspace_id, scope),
                    source_id,
                )

    def _remove_gdrive_memberships(
        self,
        records: Sequence[Record],
        scopes: Sequence[str],
    ) -> None:
        repository = self._gdrive_state_repository
        if repository is None:
            return
        for record in records:
            if not record.workspace_id:
                continue
            source_id = canonical_gdrive_source_id(record)
            for scope in scopes:
                repository.remove_membership(
                    GDriveScopeIdentity(GDRIVE_SOURCE_KIND, record.workspace_id, scope),
                    source_id,
                )

    def _recover_gdrive_replacements(self) -> None:
        recovered = False
        for entry in self._gdrive_replacement_journal.load():
            if entry.phase == "prepared":
                partial_keys = tuple(
                    key
                    for key in entry.new_keys
                    if key not in entry.old_keys
                    and self.kernel.backend.hydrate_record(key) is not None
                )
                if partial_keys:
                    self.kernel.backend.delete(list(partial_keys))
                self._gdrive_replacement_journal.complete(entry.source_key)
                recovered = True
                continue
            new_keys = tuple(
                key for key in entry.new_keys if self.kernel.backend.hydrate_record(key) is not None
            )
            self._delete_gdrive_stale_keys(entry.source_key, entry.old_keys, new_keys)
            if new_keys:
                self._source_records[entry.source_key] = list(new_keys)
            else:
                self._source_records.pop(entry.source_key, None)
            self._gdrive_replacement_journal.complete(entry.source_key)
            recovered = True
        if recovered:
            self._save_source_map()

    def reconcile_git_project_attribution(
        self,
        git_dir: Path,
        workspace_id: str | None,
    ) -> int:
        """Repair project identity on existing Git records without rebuilding them."""
        if workspace_id is None:
            return 0

        commit_ids = {
            f"git:{commit_hash}"
            for commit_hash in iter_commit_hashes_after_timestamp(git_dir)
        }
        updates: list[Record] = []
        replacements: dict[str, str] = {}
        for row in self.kernel.backend._record_rows():
            if row["source_kind"] != "git_commit":
                continue
            source_id = str(row["source_id"])
            commit_id = ":".join(source_id.split(":")[:2])
            if commit_id not in commit_ids:
                continue
            record = self.kernel.backend.hydrate_record(row["storage_key"])
            if record is None:
                continue
            metadata = dict(record.metadata)
            if (
                record.workspace_id == workspace_id
                and metadata.get("project_id") == workspace_id
            ):
                continue
            metadata["project_id"] = workspace_id
            updated = replace(record, workspace_id=workspace_id, metadata=metadata)
            updates.append(updated)
            replacements[record.storage_key] = updated.storage_key

        if not updates or not self.index_records(updates):
            return 0

        for doc_id, keys in self._source_records.items():
            replaced = [replacements.get(key, key) for key in keys]
            self._source_records[doc_id] = list(dict.fromkeys(replaced))
        stale_keys = [key for key, replacement in replacements.items() if key != replacement]
        if stale_keys:
            self.kernel.backend.delete(stale_keys)
        self._save_source_map()
        return len(updates)

    def rebuild_graph(self) -> None:
        self._rebuild_graph()

    def _rebuild_graph(self) -> None:
        edges: list[GraphEdge] = []
        for keys in self._source_records.values():
            records = tuple(
                self.kernel.backend.hydrate_record(key)
                for key in keys
            )
            hydrated = tuple(record for record in records if record is not None)
            if not hydrated:
                continue
            for record in hydrated:
                source_identities = [record.identity]
                parent_chunk_id = record.metadata.get("parent_chunk_id")
                if isinstance(parent_chunk_id, str) and parent_chunk_id:
                    source_identities.append(
                        RecordIdentity(
                            record.workspace_id,
                            record.source_kind,
                            parent_chunk_id,
                        )
                    )
                edges.extend(
                    self._graph_edges_for_document(
                        record,
                        tuple(dict.fromkeys(source_identities)),
                    )
                )
        if edges:
            try:
                self.graph.upsert_edges(edges)
            except ValueError:
                logger.debug("Skipping invalid graph edges", exc_info=True)

    def _graph_edges_for_document(
        self,
        document: Document | Record,
        source_identities: tuple[RecordIdentity, ...],
    ) -> list[GraphEdge]:
        project_id = (
            document.project_id
            if isinstance(document, Document)
            else document.workspace_id
        )
        links = document.links if isinstance(document, Document) else document.metadata.get("links", [])
        if not isinstance(links, list):
            return []
        target_ids: list[RecordIdentity] = []
        for link in links:
            if not isinstance(link, str):
                continue
            target_doc_id = self._resolve_link_doc_id(document, link)
            if target_doc_id is None:
                continue
            target_keys = self._source_records.get(target_doc_id, [])
            target_ids.extend(
                RecordIdentity.from_storage_key(key)
                for key in target_keys
            )
        return [
            GraphEdge(source, target, "links_to", 1.0)
            for source in source_identities
            for target in target_ids
            if source.workspace_id == project_id
        ]

    def _resolve_link_doc_id(self, document: Document | Record, link: str) -> str | None:
        parsed = urlparse(link)
        if parsed.scheme or parsed.netloc or link.startswith("#"):
            return None
        raw_path = unquote(parsed.path).strip()
        if not raw_path:
            return None
        normalized_path = raw_path.lstrip("/")
        source_path = (
            Path(document.file_path)
            if isinstance(document, Document)
            else Path(str(document.metadata.get("file_path", "")))
        )
        candidates = [normalized_path.removesuffix(".md")]
        if source_path:
            linked_path = (source_path.parent / raw_path).resolve()
            if linked_path.suffix.lower() == ".md":
                candidates.insert(0, self._doc_id_for_path(str(linked_path)))
            else:
                candidates.insert(0, self._doc_id_for_path(str(linked_path.with_suffix(".md"))))
        for root in self._documents_roots:
            linked_path = (root / normalized_path).resolve()
            if linked_path.suffix.lower() == ".md":
                candidates.insert(0, self._doc_id_for_path(str(linked_path)))
            else:
                candidates.insert(0, self._doc_id_for_path(str(linked_path.with_suffix(".md"))))
        for candidate in candidates:
            if candidate in self._source_records:
                return candidate
        return None

    def remove_document(self, doc_id: str) -> None:
        keys = self._source_records.pop(doc_id, [])
        if keys:
            self.kernel.backend.delete(keys)
            self._save_source_map()
            self._state_version += 1

    def persist(self) -> None:
        self._save_source_map()

    def persist_checkpoint(self) -> None:
        self.persist()

    def clear_documents(self) -> None:
        keys = [
            key
            for doc_id, record_keys in self._source_records.items()
            if not any(
                RecordIdentity.from_storage_key(key).source_kind == "git_commit"
                for key in record_keys
            )
            for key in record_keys
        ]
        if keys:
            self.kernel.backend.delete(keys)
        self._source_records = {
            doc_id: record_keys
            for doc_id, record_keys in self._source_records.items()
            if any(
                RecordIdentity.from_storage_key(key).source_kind == "git_commit"
                for key in record_keys
            )
        }
        self.persist()

    def finalize_derived_graph_state(self) -> None:
        """Compatibility hook; graph edges are written with each record batch."""
        return

    def load(self) -> None:
        self._source_records = self._load_source_map()

    def replace_vector_store(self, _vector: Any) -> None:
        raise RuntimeError("canonical record manager uses one configured embedding provider")

    def reconcile_indices(
        self,
        discovered_files: list[str],
        docs_path: Path,
        documents_roots: list[Path] | None = None,
    ) -> Any:
        del documents_roots
        discovered = {self._doc_id_for_path(path): path for path in discovered_files}
        current = set(self._source_records)
        descriptions = {
            str(description["doc_id"]): description
            for description in self.describe_documents()
            if isinstance(description.get("doc_id"), str)
        }
        added = 0
        removed = 0
        failed = 0
        for doc_id, path in discovered.items():
            if doc_id not in current:
                added += 1
                if not self.index_document(path):
                    failed += 1
        for doc_id in current - set(discovered):
            description = descriptions.get(doc_id, {})
            raw_path = description.get("file_path")
            stale_path = Path(str(raw_path)) if isinstance(raw_path, str) else None
            if stale_path is not None and stale_path.exists():
                logger.info(
                    "Removing stale entry excluded by pattern: %s",
                    stale_path,
                )
            else:
                logger.info(
                    "Removing stale entry; file missing: %s",
                    stale_path or (Path(docs_path) / doc_id),
                )
            self.remove_document(doc_id)
            removed += 1

        @dataclass(frozen=True)
        class _ReconcileResult:
            added_count: int
            removed_count: int
            moved_count: int = 0
            failed_count: int = 0

        return _ReconcileResult(added, removed, 0, failed)


def _run_async(awaitable):
    """Run ingestion from both sync workers and an active event-loop thread."""

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(awaitable)

    with ThreadPoolExecutor(max_workers=1) as executor:
        return executor.submit(asyncio.run, awaitable).result()


__all__ = ["PreparedRecordDocument", "RecordIndexManager", "build_embedding_provider"]
