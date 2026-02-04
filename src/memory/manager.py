import logging
import re
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import yaml

from src.chunking.factory import get_chunker
from src.config import Config
from src.indices.graph import GraphStore
from src.indices.keyword import KeywordIndex
from src.indices.vector import VectorIndex
from src.memory.link_parser import extract_links
from src.memory.models import ExtractedLink, MemoryDocument, MemoryFrontmatter
from src.memory.storage import (
    compute_memory_id,
    ensure_memory_dirs,
    get_memory_file_path,
    get_indices_path,
    list_memory_files,
)
from src.models import Document

logger = logging.getLogger(__name__)


FRONTMATTER_PATTERN = re.compile(r'^---\s*\n(.*?)\n---\s*\n', re.DOTALL)


@dataclass
class FailedMemory:
    path: str
    error: str
    timestamp: str


@dataclass
class MemoryMetadata:
    """Cached metadata for a single memory file."""
    memory_id: str
    tags: list[str]
    memory_type: str
    size_bytes: int


class MemoryIndexManager:
    """Manages memory file indexing and persistence.

    OWNERSHIP: Memory indices are owned by the main process, not the worker.
    In multiprocess mode, the worker handles document indexing via snapshots,
    but memory operations are handled directly by the main process.

    Index Storage: Indices are stored at `memory_path/indices/`, NOT in the
    document snapshot directory. This is intentional - memories don't
    participate in snapshot-based sync.

    Typical lifecycle:
        1. Create manager with indices (empty or loaded)
        2. Call load() to restore persisted indices
        3. Call reconcile() to detect new/modified files
        4. Call persist() if reconcile() made changes
        5. Use index_memory()/remove_memory() for incremental updates
    """

    def __init__(
        self,
        config: Config,
        memory_path: Path,
        vector: VectorIndex,
        keyword: KeywordIndex,
        graph: GraphStore,
    ):
        self._config = config
        self._memory_path = memory_path
        self._vector = vector
        self._keyword = keyword
        self._graph = graph
        self._failed_files: list[FailedMemory] = []
        self._chunker = get_chunker(config.memory_chunking)

        # Checkpoint tracking (protected by _checkpoint_lock)
        self._checkpoint_lock = threading.Lock()
        self._ops_since_checkpoint: int = 0
        self._last_checkpoint_time: float = time.time()
        self._dirty: bool = False

        # Metadata cache for fast stats queries (protected by _cache_lock)
        self._cache_lock = threading.Lock()
        self._metadata_cache: dict[str, MemoryMetadata] = {}

        ensure_memory_dirs(memory_path)

    @property
    def memory_path(self) -> Path:
        return self._memory_path

    @property
    def vector(self) -> VectorIndex:
        return self._vector

    @property
    def keyword(self) -> KeywordIndex:
        return self._keyword

    @property
    def graph(self) -> GraphStore:
        return self._graph

    @property
    def is_dirty(self) -> bool:
        """Return True if there are unsaved changes."""
        return self._dirty

    def _maybe_checkpoint(self) -> None:
        """Persist indices if checkpoint threshold is reached.

        Checkpointing occurs when either:
        - N operations have been performed since last checkpoint
        - M seconds have elapsed since last checkpoint

        This provides crash resilience by periodically persisting changes.
        Lock scope is minimal: state checks/updates under lock, I/O outside.
        """
        should_checkpoint = False

        with self._checkpoint_lock:
            if not self._dirty:
                return

            self._ops_since_checkpoint += 1
            now = time.time()

            should_checkpoint = (
                self._ops_since_checkpoint >= self._config.memory.checkpoint_interval_ops
                or (now - self._last_checkpoint_time) >= self._config.memory.checkpoint_interval_secs
            )

            if should_checkpoint:
                self._ops_since_checkpoint = 0
                self._last_checkpoint_time = now
                self._dirty = False

        if should_checkpoint:
            self._persist_indices()
            logger.debug("Memory indices checkpointed")

    def _parse_frontmatter(self, content: str) -> tuple[MemoryFrontmatter, str]:
        match = FRONTMATTER_PATTERN.match(content)
        if not match:
            return MemoryFrontmatter(), content

        try:
            yaml_content = match.group(1)
            data = yaml.safe_load(yaml_content) or {}

            created_at = data.get("created_at")
            if isinstance(created_at, str):
                try:
                    created_at = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
                except ValueError:
                    created_at = None
            elif isinstance(created_at, datetime):
                pass
            else:
                created_at = None

            tags = data.get("tags", [])
            if isinstance(tags, str):
                tags = [t.strip() for t in tags.split(",") if t.strip()]

            frontmatter = MemoryFrontmatter(
                type=data.get("type", "journal"),
                status=data.get("status", "active"),
                tags=tags,
                created_at=created_at,
            )

            body = content[match.end():]
            return frontmatter, body

        except yaml.YAMLError as e:
            logger.warning(f"Failed to parse frontmatter: {e}")
            return MemoryFrontmatter(), content

    def _parse_memory_file(self, file_path: Path) -> MemoryDocument:
        content = file_path.read_text(encoding="utf-8")
        frontmatter, body = self._parse_frontmatter(content)
        links = extract_links(body)
        memory_id = compute_memory_id(self._memory_path, file_path)
        stat = file_path.stat()
        modified_time = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc)

        return MemoryDocument(
            id=memory_id,
            content=body,
            frontmatter=frontmatter,
            links=links,
            file_path=str(file_path),
            modified_time=modified_time,
        )

    def _normalize_memory_name(self, memory_id: str):
        if memory_id.startswith("memory:"):
            return memory_id.split("memory:", 1)[1]
        return memory_id

    def _resolve_memory_path(self, memory_id: str):
        name = self._normalize_memory_name(memory_id)
        candidate = get_memory_file_path(self._memory_path, name)
        if candidate.exists():
            return candidate
        return None

    def _memory_to_document(self, memory: MemoryDocument) -> Document:
        metadata: dict[str, str | list[str] | int | float | bool] = {
            "memory_type": memory.frontmatter.type,
            "memory_status": memory.frontmatter.status,
        }

        if memory.frontmatter.created_at:
            metadata["created_at"] = memory.frontmatter.created_at.isoformat()

        link_targets = [link.target for link in memory.links]

        return Document(
            id=memory.id,
            content=memory.content,
            metadata=metadata,
            links=link_targets,
            tags=memory.frontmatter.tags,
            file_path=memory.file_path,
            modified_time=memory.modified_time,
        )

    def _add_tag_nodes_and_edges(self, memory_id: str, tags: list[str]) -> None:
        from src.memory.link_parser import normalize_tag

        for tag in tags:
            normalized_tag = normalize_tag(tag)
            tag_id = f"tag:{normalized_tag}"

            if not self._graph.has_node(tag_id):
                self._graph.add_node(tag_id, {"is_tag": True, "tag_name": normalized_tag})

            self._graph.add_edge(
                source=memory_id,
                target=tag_id,
                edge_type="HAS_TAG",
                edge_context="",
            )

    def _add_ghost_nodes_and_edges(
        self, memory_id: str, links: list[ExtractedLink]
    ) -> None:
        for link in links:
            if link.is_memory_link:
                target_id = link.target
            else:
                target_id = f"ghost:{link.target}"

            if not self._graph.has_node(target_id):
                if link.is_memory_link:
                    self._graph.add_node(target_id, {"is_memory_ghost": True})
                else:
                    self._graph.add_node(target_id, {"is_ghost": True, "target": link.target})

            self._graph.add_edge(
                source=memory_id,
                target=target_id,
                edge_type=link.edge_type,
                edge_context=link.anchor_context,
            )

    def index_memory(self, file_path: str) -> None:
        path = Path(file_path)
        try:
            memory = self._parse_memory_file(path)
            document = self._memory_to_document(memory)

            chunks = self._chunker.chunk_document(document)

            for chunk in chunks:
                chunk.metadata["memory_type"] = memory.frontmatter.type
                chunk.metadata["memory_status"] = memory.frontmatter.status
                chunk.metadata["memory_tags"] = memory.frontmatter.tags
                if memory.frontmatter.created_at:
                    chunk.metadata["memory_created_at"] = memory.frontmatter.created_at.isoformat()

                self._vector.add_chunk(chunk)
                self._keyword.add_chunk(chunk)

            self._graph.add_node(memory.id, {
                "type": memory.frontmatter.type,
                "status": memory.frontmatter.status,
                "tags": memory.frontmatter.tags,
            })

            self._add_tag_nodes_and_edges(memory.id, memory.frontmatter.tags)
            self._add_ghost_nodes_and_edges(memory.id, memory.links)

            self._failed_files = [
                f for f in self._failed_files if f.path != file_path
            ]

            # Update metadata cache
            with self._cache_lock:
                self._metadata_cache[memory.id] = MemoryMetadata(
                    memory_id=memory.id,
                    tags=list(memory.frontmatter.tags),
                    memory_type=memory.frontmatter.type,
                    size_bytes=path.stat().st_size,
                )

            with self._checkpoint_lock:
                self._dirty = True
            self._maybe_checkpoint()

            logger.info(f"Indexed memory: {memory.id} with {len(chunks)} chunks")

        except Exception as e:
            logger.error(f"Failed to index memory {file_path}: {e}", exc_info=True)
            failed = FailedMemory(
                path=file_path,
                error=str(e),
                timestamp=datetime.now(timezone.utc).isoformat(),
            )
            self._failed_files = [
                f for f in self._failed_files if f.path != file_path
            ] + [failed]
            raise

    def remove_memory(self, memory_id: str) -> None:
        try:
            self._vector.remove(memory_id)
            self._keyword.remove(memory_id)
            self._graph.remove_node(memory_id)

            # Remove from metadata cache
            with self._cache_lock:
                self._metadata_cache.pop(memory_id, None)

            with self._checkpoint_lock:
                self._dirty = True
            self._maybe_checkpoint()
            logger.info(f"Removed memory: {memory_id}")
        except Exception as e:
            logger.error(f"Failed to remove memory {memory_id}: {e}", exc_info=True)

    def reindex_memory(self, memory_id: str, reason: str | None = None):
        file_path = self._resolve_memory_path(memory_id)
        if not file_path:
            if reason:
                logger.warning("Reindex skipped for %s (reason: %s): file not found", memory_id, reason)
            else:
                logger.warning("Reindex skipped for %s: file not found", memory_id)
            return False

        try:
            self.remove_memory(memory_id)
            self.index_memory(str(file_path))
            if reason:
                logger.info("Reindexed %s from %s (reason: %s)", memory_id, file_path, reason)
            else:
                logger.info("Reindexed %s from %s", memory_id, file_path)
            return True
        except Exception as e:
            logger.error("Failed to reindex %s: %s", memory_id, e, exc_info=True)
            return False

    def reindex_all(self) -> int:
        memory_files = list_memory_files(self._memory_path)
        indexed_count = 0

        for file_path in memory_files:
            try:
                self.index_memory(str(file_path))
                indexed_count += 1
            except Exception as e:
                logger.error(f"Failed to index {file_path}: {e}")

        return indexed_count

    def _persist_indices(self) -> None:
        """Internal: persist indices without touching dirty flag.

        Used by _maybe_checkpoint() which manages dirty flag under lock.
        """
        indices_path = get_indices_path(self._memory_path)
        self._vector.persist(indices_path / "vector")
        self._keyword.persist(indices_path / "keyword")
        self._graph.persist(indices_path / "graph")
        logger.info(f"Persisted memory indices to {indices_path}")

    def persist(self) -> None:
        """Persist indices to disk and clear dirty flag.

        Public API for explicit persistence (e.g., on shutdown).
        """
        try:
            self._persist_indices()
            with self._checkpoint_lock:
                self._dirty = False
        except Exception as e:
            logger.error(f"Failed to persist memory indices: {e}", exc_info=True)
            raise

    def load(self) -> None:
        indices_path = get_indices_path(self._memory_path)
        try:
            self._vector.load(indices_path / "vector")
            self._keyword.load(indices_path / "keyword")
            self._graph.load(indices_path / "graph")
            self._rebuild_metadata_cache()
            logger.info(f"Loaded memory indices from {indices_path}")
        except Exception as e:
            logger.error(f"Failed to load memory indices: {e}", exc_info=True)
            raise

    def _rebuild_metadata_cache(self) -> None:
        """Rebuild metadata cache from filesystem.

        Called on load() to populate cache from memory files.
        """
        cache: dict[str, MemoryMetadata] = {}
        for memory_file in list_memory_files(self._memory_path):
            try:
                memory_id = compute_memory_id(self._memory_path, memory_file)
                content = memory_file.read_text(encoding="utf-8")
                frontmatter, _ = self._parse_frontmatter(content)
                cache[memory_id] = MemoryMetadata(
                    memory_id=memory_id,
                    tags=list(frontmatter.tags),
                    memory_type=frontmatter.type,
                    size_bytes=memory_file.stat().st_size,
                )
            except Exception:
                continue
        with self._cache_lock:
            self._metadata_cache = cache
        logger.debug(f"Rebuilt metadata cache with {len(cache)} entries")

    def reconcile(self) -> int:
        memory_files = list_memory_files(self._memory_path)
        indexed_ids = set(self._vector.get_document_ids())

        reindexed_count = 0
        for file_path in memory_files:
            memory_id = compute_memory_id(self._memory_path, file_path)

            if memory_id not in indexed_ids:
                try:
                    self.index_memory(str(file_path))
                    reindexed_count += 1
                except Exception as e:
                    logger.warning(f"Failed to reconcile memory {file_path}: {e}")
                continue

            chunk_ids = self._vector.get_chunk_ids_for_document(memory_id)
            if not chunk_ids:
                try:
                    self.index_memory(str(file_path))
                    reindexed_count += 1
                except Exception as e:
                    logger.warning(f"Failed to reconcile memory {file_path}: {e}")
                continue

            chunk_data = self._vector.get_chunk_by_id(chunk_ids[0])
            if not chunk_data:
                continue

            metadata = chunk_data.get("metadata", {})
            if not isinstance(metadata, dict):
                continue

            if "memory_type" not in metadata or "memory_created_at" not in metadata:
                logger.info(f"Reindexing memory with missing metadata: {memory_id}")
                try:
                    self.remove_memory(memory_id)
                    self.index_memory(str(file_path))
                    reindexed_count += 1
                except Exception as e:
                    logger.warning(f"Failed to reindex memory {file_path}: {e}")

        if reindexed_count > 0:
            logger.info(f"Reconciled {reindexed_count} memories with missing metadata")
            self._maybe_checkpoint()

        return reindexed_count

    def get_memory_count(self) -> int:
        return len(self._vector.get_document_ids())

    def get_failed_files(self) -> list[dict[str, str]]:
        return [
            {"path": f.path, "error": f.error, "timestamp": f.timestamp}
            for f in self._failed_files
        ]

    def get_all_tags(self) -> dict[str, int]:
        with self._cache_lock:
            tag_counts: dict[str, int] = {}
            for metadata in self._metadata_cache.values():
                for tag in metadata.tags:
                    tag_counts[tag] = tag_counts.get(tag, 0) + 1
            return tag_counts

    def get_all_types(self) -> dict[str, int]:
        with self._cache_lock:
            type_counts: dict[str, int] = {}
            for metadata in self._metadata_cache.values():
                type_counts[metadata.memory_type] = type_counts.get(metadata.memory_type, 0) + 1
            return type_counts

    def get_total_size_bytes(self) -> int:
        with self._cache_lock:
            return sum(m.size_bytes for m in self._metadata_cache.values())
