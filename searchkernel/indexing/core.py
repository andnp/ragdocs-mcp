"""Pure chunking/embedding/indexing transforms over Record/Chunk/vectors.

`IndexCore` owns the source-agnostic per-document indexing sequence: chunk a
document, decide delta vs. full re-index, and mutate the vector/keyword/graph
indices accordingly. It has no filesystem-watching, daemon, or bootstrap
coupling — `IndexManager` (in `manager.py`) wraps it with file discovery,
manifest tracking, and task-submission concerns.
"""

import logging
from collections.abc import Callable

from searchkernel.domain import Record
from searchkernel.indices.graph import GraphStore
from searchkernel.indices.hash_store import ChunkHashStore
from searchkernel.indices.keyword import KeywordIndex
from searchkernel.indices.vector import VectorIndex
from searchkernel.models import Chunk, Document
from searchkernel.pipeline.stage import SearchContext
from searchkernel.pipeline.stages.chunk import ChunkStage
from searchkernel.pipeline.stages.index import IndexStage

logger = logging.getLogger(__name__)


class IndexCore:
    def __init__(
        self,
        chunker,
        vector: VectorIndex,
        keyword: KeywordIndex,
        graph: GraphStore,
        hash_store: ChunkHashStore,
    ):
        self._chunker = chunker
        self.vector = vector
        self.keyword = keyword
        self.graph = graph
        self._hash_store = hash_store

    def chunk_document(self, document: Document) -> list[Chunk]:
        context = ChunkStage(self._chunker).run(
            SearchContext(query="", metadata={"document": document})
        )
        return context.metadata["chunks"]

    def index_chunks(self, chunks: list[Chunk]) -> None:
        IndexStage(self.vector, self.keyword, self.graph).run(
            SearchContext(query="", metadata={"chunks": chunks})
        )

    def detect_changed_chunks(
        self, chunks: list[Chunk]
    ) -> tuple[list[Chunk], list[str]]:
        """Identify chunks with changed content.

        Returns:
            (changed_chunks, unchanged_chunk_ids)
        """
        changed = []
        unchanged = []

        for chunk in chunks:
            if self._hash_store.has_changed(chunk):
                changed.append(chunk)
            else:
                unchanged.append(chunk.chunk_id)

        return changed, unchanged

    def should_use_delta_indexing(
        self,
        changed_chunks: list[Chunk],
        total_chunks: int,
        threshold: float,
    ) -> bool:
        """Decide whether to use delta or full re-index based on change ratio."""
        if total_chunks == 0:
            return True

        change_ratio = len(changed_chunks) / total_chunks

        if change_ratio > threshold:
            logger.info(
                f"Change ratio {change_ratio:.1%} exceeds threshold {threshold:.1%}, "
                "using full re-index"
            )
            return False

        return True

    def update_chunks(self, doc_id: str, chunks: list[Chunk]) -> None:
        """Update specific chunks in all indices (remove old → add new)."""
        if not chunks:
            return

        chunk_ids = [chunk.chunk_id for chunk in chunks]

        # Remove old versions (batch where possible)
        for chunk_id in chunk_ids:
            self.vector.remove_chunk(chunk_id)
        self.keyword.remove_chunks(chunk_ids)
        for chunk_id in chunk_ids:
            self.graph.remove_chunk(chunk_id)

        # Add new versions
        self.index_chunks(chunks)

        logger.debug(f"Updated {len(chunks)} chunks for {doc_id}")

    def full_reindex_document(
        self,
        doc_id: str,
        chunks: list[Chunk],
        on_persist_hash_store: Callable[[], None] | None = None,
    ) -> None:
        """Full re-index of document (remove all old chunks, add all new)."""
        # Remove all old chunks
        self.vector.remove(doc_id)
        self.keyword.remove(doc_id)
        self.graph.remove_node(doc_id)

        # Add all new chunks
        self.index_chunks(chunks)

        # Update hash store (clear old hashes first)
        self._hash_store.remove_document(doc_id)
        for chunk in chunks:
            self._hash_store.set_hash(chunk.chunk_id, chunk.content_hash)
        if on_persist_hash_store is not None:
            on_persist_hash_store()

        logger.debug(f"Full re-indexed {doc_id} with {len(chunks)} chunks")

    def index_record(
        self,
        record: Record,
        on_persist_hash_store: Callable[[], None] | None = None,
    ) -> bool:
        """Ingest a source-agnostic Record (e.g. a git commit) into the live indices.

        Unlike chunk_document/index_chunks over a file-backed Document, this
        chunks in-memory content (record.body) rather than reading a file from
        disk. Every chunk is tagged with source_kind/source_id metadata so
        SearchOrchestrator.query's source_filter can scope a query to this
        source.

        Returns:
            True if the indices were mutated, False if the record was
            unchanged and skipped.
        """
        document = Document(
            id=record.source_id,
            content=record.body,
            metadata={**record.metadata, "title": record.title},
            links=[],
            tags=[],
            file_path=record.uri or "",
            modified_time=record.updated_at,
        )

        chunks = self.chunk_document(document)
        for chunk in chunks:
            chunk.metadata = {
                **chunk.metadata,
                "source_kind": record.source_kind,
                "source_id": record.source_id,
            }

        changed_chunks, _unchanged_chunk_ids = self.detect_changed_chunks(chunks)
        if not changed_chunks and document.id in self.vector.get_document_ids():
            logger.debug("No changes in record %s, skipping re-index", record.source_id)
            return False

        self.full_reindex_document(
            document.id, chunks, on_persist_hash_store=on_persist_hash_store
        )

        graph_metadata = {**document.metadata, "source_kind": record.source_kind}
        self.graph.add_node(document.id, graph_metadata)

        return True
