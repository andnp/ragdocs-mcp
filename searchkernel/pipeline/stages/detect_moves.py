"""DetectMovesStage: file-move detection ingestion stage.

Lifted from `IndexManager._detect_file_moves`, the first half of the
dedup/canonicalize phase of the ingestion path (discover -> chunk ->
embed -> index -> dedup/canonicalize -> re-embed/repair). Pure
delegate to the same content-hash-overlap comparison `IndexManager`
used inline during `reconcile_indices` -- same inputs, same outputs.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Protocol

from searchkernel.domain import Chunk
from searchkernel.pipeline.stage import SearchContext

_REMOVED_DOC_IDS_KEY = "removed_doc_ids"
_ADDED_DOCS_KEY = "added_docs"
_MOVE_THRESHOLD_KEY = "move_detection_threshold"
_MOVED_FILES_KEY = "moved_files"


class _HashLookup(Protocol):
    def get_chunks_by_document(self, doc_id: str) -> list[tuple[str, str]] | None: ...


class DetectMovesStage:
    """Detect file moves by comparing chunk content-hash overlap.

    Expects `context.metadata["removed_doc_ids"]` (set[str]),
    `context.metadata["added_docs"]` (dict[str, list[Chunk]]) and
    `context.metadata["move_detection_threshold"]` (float). Writes
    `context.metadata["moved_files"]` (dict[str, str], old_doc_id ->
    new_doc_id) for the detected moves.
    """

    name = "detect_moves"

    def __init__(self, hash_store: _HashLookup):
        self._hash_store = hash_store

    def run(self, context: SearchContext) -> SearchContext:
        removed_docs: set[str] = context.metadata[_REMOVED_DOC_IDS_KEY]
        added_docs: dict[str, list[Chunk]] = context.metadata[_ADDED_DOCS_KEY]
        threshold: float = context.metadata[_MOVE_THRESHOLD_KEY]

        moves: dict[str, str] = {}

        for new_doc_id, new_chunks in added_docs.items():
            if not new_chunks:
                continue

            new_hashes = {chunk.content_hash for chunk in new_chunks}

            best_match_doc = None
            best_match_ratio = 0.0

            for old_doc_id in removed_docs:
                old_chunk_data = self._hash_store.get_chunks_by_document(old_doc_id)
                if not old_chunk_data:
                    continue

                old_hashes = {hash_val for _, hash_val in old_chunk_data}
                if not old_hashes or not new_hashes:
                    continue

                matching_hashes = new_hashes & old_hashes
                match_ratio = len(matching_hashes) / max(len(old_hashes), len(new_hashes))

                if match_ratio > best_match_ratio:
                    best_match_ratio = match_ratio
                    best_match_doc = old_doc_id

            if best_match_doc and best_match_ratio >= threshold:
                moves[best_match_doc] = new_doc_id

        metadata = dict(context.metadata)
        metadata[_MOVED_FILES_KEY] = moves
        return replace(context, metadata=metadata)
