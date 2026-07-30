"""ApplyMoveStage: file-move application ingestion stage.

Lifted from `IndexManager._apply_file_move`, the second half of the
dedup/canonicalize phase of the ingestion path (discover -> chunk ->
embed -> index -> dedup/canonicalize -> re-embed/repair). Renames a
detected move's chunks across vector/keyword/graph/hash-store instead
of removing and fully re-indexing/re-embedding them.

Persistence (`ChunkHashStore.persist`), logging, and the manager's
derived-graph-state/state-version bookkeeping stay at the call site
(`IndexManager._apply_file_move`), matching `IndexStage`'s precedent
of leaving those manager-level concerns outside the stage.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Protocol

from searchkernel.domain import Chunk
from searchkernel.pipeline.stage import SearchContext

_OLD_DOC_ID_KEY = "old_doc_id"
_NEW_DOC_ID_KEY = "new_doc_id"
_NEW_CHUNKS_KEY = "new_chunks"
_MOVE_APPLIED_KEY = "move_applied"
_MOVED_CHUNK_COUNT_KEY = "moved_chunk_count"
_HASH_STORE_UPDATED_KEY = "hash_store_updated"


class _VectorMover(Protocol):
    def update_chunk_path(
        self, old_chunk_id: str, new_chunk_id: str, new_metadata: dict
    ) -> bool: ...


class _KeywordMover(Protocol):
    def move_chunk(self, old_chunk_id: str, new_chunk: Chunk) -> bool: ...


class _GraphMover(Protocol):
    def rename_node(self, old_chunk_id: str, new_chunk_id: str) -> bool: ...


class _HashMover(Protocol):
    def get_chunks_by_document(self, doc_id: str) -> list[tuple[str, str]] | None: ...
    def remove_document(self, doc_id: str) -> None: ...
    def set_hash(self, chunk_id: str, content_hash: str) -> None: ...


class ApplyMoveStage:
    """Rename a moved document's chunks across indices without re-embedding.

    Expects `context.metadata["old_doc_id"]`, `["new_doc_id"]`, and
    `["new_chunks"]` (`list[Chunk]`). Writes
    `context.metadata["move_applied"]` (bool -- `False` means the
    caller should fall back to full re-index), `["moved_chunk_count"]`
    (int), and `["hash_store_updated"]` (bool -- `False` only when no
    stored hashes existed for `old_doc_id`, meaning the hash store was
    never touched).
    """

    name = "apply_move"

    def __init__(
        self,
        vector: _VectorMover,
        keyword: _KeywordMover,
        graph: _GraphMover,
        hash_store: _HashMover,
        move_detection_threshold: float,
    ):
        self._vector = vector
        self._keyword = keyword
        self._graph = graph
        self._hash_store = hash_store
        self._threshold = move_detection_threshold

    def run(self, context: SearchContext) -> SearchContext:
        old_doc_id: str = context.metadata[_OLD_DOC_ID_KEY]
        new_doc_id: str = context.metadata[_NEW_DOC_ID_KEY]
        new_chunks: list[Chunk] = context.metadata[_NEW_CHUNKS_KEY]

        applied, moved_count, hash_store_updated = self._apply(
            old_doc_id, new_doc_id, new_chunks
        )

        metadata = dict(context.metadata)
        metadata[_MOVE_APPLIED_KEY] = applied
        metadata[_MOVED_CHUNK_COUNT_KEY] = moved_count
        metadata[_HASH_STORE_UPDATED_KEY] = hash_store_updated
        return replace(context, metadata=metadata)

    def _apply(
        self, old_doc_id: str, new_doc_id: str, new_chunks: list[Chunk]
    ) -> tuple[bool, int, bool]:
        old_chunk_data = self._hash_store.get_chunks_by_document(old_doc_id)
        if not old_chunk_data:
            return False, 0, False

        old_hash_to_chunk = {
            hash_val: chunk_id for chunk_id, hash_val in old_chunk_data
        }

        moved_count = 0

        for new_chunk in new_chunks:
            old_chunk_id = old_hash_to_chunk.get(new_chunk.content_hash)
            if not old_chunk_id:
                continue

            new_metadata = {
                "doc_id": new_chunk.record_id,
                "chunk_id": new_chunk.chunk_id,
                "file_path": new_chunk.metadata.get("file_path", ""),
                "header_path": new_chunk.metadata.get("header_path", ""),
                **new_chunk.metadata,
            }

            if not self._vector.update_chunk_path(
                old_chunk_id, new_chunk.chunk_id, new_metadata
            ):
                continue

            if not self._keyword.move_chunk(old_chunk_id, new_chunk):
                continue

            self._graph.rename_node(old_chunk_id, new_chunk.chunk_id)
            moved_count += 1

        self._hash_store.remove_document(old_doc_id)
        for chunk in new_chunks:
            self._hash_store.set_hash(chunk.chunk_id, chunk.content_hash)

        success_ratio = moved_count / len(new_chunks)
        applied = success_ratio >= self._threshold

        return applied, moved_count, True
