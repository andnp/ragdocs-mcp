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
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from searchkernel.api import (
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
from mcp_markdown_ragdocs.models import Document
from mcp_markdown_ragdocs.parsers.dispatcher import dispatch_parser

logger = logging.getLogger(__name__)


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

    def neighbors_many(self, identities, *, depth: int, max_neighbors: int | None = None):
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
        identities,
        *,
        depth: int,
        max_neighbors: int | None = None,
    ):
        return self._graph_store.neighbors_many(
            identities,
            depth=depth,
            max_neighbors=max_neighbors,
        )

    def neighbors(self, identity, *, depth: int, max_neighbors: int | None = None):
        return self.neighbors_many(
            [identity],
            depth=depth,
            max_neighbors=max_neighbors,
        )[identity.storage_key]

    def incoming_neighbors(self, identity, *, depth: int, max_neighbors: int | None = None):
        return self.incoming_neighbors_many(
            [identity],
            depth=depth,
            max_neighbors=max_neighbors,
        )[identity.storage_key]

    def incoming_neighbors_many(
        self,
        identities,
        *,
        depth: int,
        max_neighbors: int | None = None,
    ):
        requested = {identity.storage_key for identity in identities}
        incoming = {identity.storage_key: [] for identity in identities}
        outgoing = self._graph_store.neighbors_many(
            self._identities(),
            depth=depth,
            max_neighbors=None,
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
        return incoming


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
        del force
        indexed = False
        for file_path in file_paths:
            indexed = self.index_document(
                file_path,
                update_graph=False,
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
        del force
        try:
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
        try:
            receipt = _run_async(self.ingestor.index_records(records))
            if receipt.failed:
                return False
            for record in records:
                self._source_records.setdefault(record.source_id, []).append(
                    record.storage_key
                )
            self._save_source_map()
            self._state_version += 1
            return True
        except Exception:
            logger.exception("Failed to index record batch")
            return False

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
            edges.extend(
                self._graph_edges_for_document(
                    hydrated[0],
                    tuple(record.identity for record in hydrated),
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
