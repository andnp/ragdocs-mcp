"""Canonical record indexing for Markdown sources.

The application owns file discovery and parsing.  Searchkernel owns durable
record storage, embedding, and retrieval.  This module is the small seam that
connects those two responsibilities.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import re
from collections.abc import Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from searchkernel.api import (
    ContentSource,
    GraphEdge,
    LocalRecordKernel,
    OllamaEmbeddingProvider,
    Record,
    RecordIdentity,
    SemanticRecordIngestor,
    SQLiteEmbeddingCache,
    Vector,
    compute_doc_id,
    compute_doc_id_multi_root,
    get_chunker,
)

from mcp_markdown_ragdocs.config import Config
from mcp_markdown_ragdocs.git.repository import iter_commit_hashes_after_timestamp
from mcp_markdown_ragdocs.gdrive.replacement import (
    GDriveReplacementJournal,
    REPLACEMENT_JOURNAL_FILENAME,
)
from mcp_markdown_ragdocs.gdrive.replacement_policy import GDriveReplacementPolicy
from mcp_markdown_ragdocs.gdrive.records import SOURCE_KIND as GDRIVE_SOURCE_KIND
from mcp_markdown_ragdocs.gdrive.adapter import GDriveStateRepository
from mcp_markdown_ragdocs.models import Document
from mcp_markdown_ragdocs.indexing.markdown_documents import (
    MarkdownDocumentPlanner,
    SemanticDocumentWriter,
)
from mcp_markdown_ragdocs.indexing.local_graph import install_bidirectional_graph_store
from mcp_markdown_ragdocs.indexing.record_ports import (
    JsonSourceMapStore,
    DocumentPlanner,
    DocumentWriter,
    GraphCapability,
    LocalRecordStorage,
    PreparedRecordDocument,
    RecordStorage,
    SourceMapStore,
)

logger = logging.getLogger(__name__)
_FILE_MTIME_METADATA_KEY = "_file_mtime_ns"
_FILE_SIZE_METADATA_KEY = "_file_size"


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
        storage: RecordStorage | None = None,
        graph: GraphCapability | None = None,
        source_map_store: SourceMapStore | None = None,
        document_planner: DocumentPlanner | None = None,
        document_writer: DocumentWriter | None = None,
    ) -> None:
        self._config = config
        self.kernel = kernel
        self.storage = storage or LocalRecordStorage(kernel)
        self._graph = graph or (
            self.storage.graph
            if isinstance(self.storage, LocalRecordStorage)
            else kernel.graph_store
        )
        self.embedding_provider = embedding_provider
        self._documents_roots = [
            root.resolve()
            for root in (documents_roots or [Path(config.indexing.documents_path)])
        ]
        self._document_planner = document_planner or MarkdownDocumentPlanner(
            config,
            self._documents_roots,
            get_chunker(config.chunking),
        )
        self._failed_files: list[dict[str, str]] = []
        self._state_version = 0
        self._ready = True
        self._source_map_store = source_map_store or JsonSourceMapStore(
            Path(config.indexing.index_path) / "record-sources.json"
        )
        self._source_records: dict[str, list[str]] = self._source_map_store.load()
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
        self._document_writer = document_writer or SemanticDocumentWriter(
            self.ingestor,
            self.storage,
        )
        self._gdrive_replacement_policy = GDriveReplacementPolicy(
            self.ingestor,
            self.storage,
            self._source_records,
            self._source_map_store,
            self._gdrive_replacement_journal,
            self._gdrive_state_repository,
        )
        self._gdrive_replacement_policy.recover()

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
        return self._graph

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
        self.storage.register_content_source(source)
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
            records = [self.storage.hydrate_record(key) for key in keys]
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
        record = self.storage.hydrate_record(keys[0])
        if record is None:
            return False
        return (
            record.metadata.get(_FILE_MTIME_METADATA_KEY) == file_stat.st_mtime_ns
            and record.metadata.get(_FILE_SIZE_METADATA_KEY) == file_stat.st_size
        )

    def _save_source_map(self) -> None:
        self._source_map_store.save(self._source_records)

    def prepare_document(self, file_path: str) -> PreparedRecordDocument:
        return self._document_planner.plan(file_path)

    async def _index_prepared(
        self,
        prepared: PreparedRecordDocument,
        *,
        update_graph: bool = True,
    ) -> None:
        old_keys = self._source_records.get(prepared.document.id, [])
        old_records = (
            tuple(
                record
                for key in old_keys
                if (record := self.storage.hydrate_record(key)) is not None
            )
            if update_graph
            else ()
        )
        new_keys = await self._document_writer.write(prepared, old_keys)
        self._source_records[prepared.document.id] = list(new_keys)
        if update_graph:
            self._update_document_graph(old_records, new_keys)
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
            _run_async(self._gdrive_replacement_policy.replace(records))
            self._rebuild_graph()
            self._state_version += 1
            return True
        except Exception as error:
            logger.exception("Failed to index Google Drive record batch")
            self._failed_files.append({"path": "gdrive", "error": str(error)})
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
        for record in self.storage.iter_records():
            if record.source_kind != "git_commit":
                continue
            source_id = record.source_id
            commit_id = ":".join(source_id.split(":")[:2])
            if commit_id not in commit_ids:
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
            self.storage.delete(stale_keys)
        self._save_source_map()
        return len(updates)

    def rebuild_graph(self) -> None:
        self._rebuild_graph()

    def _rebuild_graph(self) -> None:
        edges: list[GraphEdge] = []
        for keys in self._source_records.values():
            records = tuple(
                self.storage.hydrate_record(key)
                for key in keys
            )
            hydrated = tuple(record for record in records if record is not None)
            if not hydrated:
                continue
            for record in hydrated:
                edges.extend(
                    self._graph_edges_for_document(
                        record,
                        self._source_identities_for_record(record),
                    )
                )
        if edges:
            try:
                self.graph.upsert_edges(edges)
            except ValueError:
                logger.debug("Skipping invalid graph edges", exc_info=True)

    def _source_identities_for_record(
        self,
        record: Record,
    ) -> tuple[RecordIdentity, ...]:
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
        return tuple(dict.fromkeys(source_identities))

    def _update_document_graph(
        self,
        old_records: Sequence[Record],
        new_keys: Sequence[str],
    ) -> None:
        """Recompute graph edges for one just-indexed document.

        Full rebuilds hydrate and re-derive edges for every document in the
        index; this instead touches only the source identities this one
        document owned before and after the edit, so indexing N documents
        does not cost O(N^2) hydrations.
        """
        new_records = tuple(
            record
            for key in new_keys
            if (record := self.storage.hydrate_record(key)) is not None
        )
        stale_identities: list[RecordIdentity] = []
        for record in (*old_records, *new_records):
            stale_identities.extend(self._source_identities_for_record(record))
        stale_identities = list(dict.fromkeys(stale_identities))

        if stale_identities:
            try:
                existing = self.graph.neighbors_many(stale_identities, depth=1)
            except ValueError:
                existing = {}
            old_edges = [
                GraphEdge(
                    RecordIdentity.from_storage_key(source_key),
                    neighbor.identity,
                    neighbor.edge_type,
                    neighbor.weight,
                )
                for source_key, neighbors in existing.items()
                for neighbor in neighbors
            ]
            if old_edges:
                try:
                    self.graph.delete_edges(old_edges)
                except ValueError:
                    logger.debug("Skipping invalid graph edge deletion", exc_info=True)

        new_edges: list[GraphEdge] = []
        for record in new_records:
            new_edges.extend(
                self._graph_edges_for_document(
                    record,
                    self._source_identities_for_record(record),
                )
            )
        if new_edges:
            try:
                self.graph.upsert_edges(new_edges)
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
            self.storage.delete(keys)
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
            self.storage.delete(keys)
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
        self._source_records = self._source_map_store.load()

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


__all__ = [
    "PreparedRecordDocument",
    "RecordIndexManager",
    "build_embedding_provider",
    "install_bidirectional_graph_store",
]
