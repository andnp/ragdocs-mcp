"""IndexStage: writes chunks into the vector/keyword/graph indices.

Lifted from the identical "add new chunks" block that `IndexManager`
duplicated in both `_update_chunks` (delta re-index) and
`_full_reindex_document` (full re-index): `vector.add_chunks` +
`keyword.add_chunks` + one `graph.add_node` per chunk. Fourth phase of
the ingestion path (discover -> chunk -> embed -> index ->
dedup/canonicalize -> re-embed/repair) -- embedding itself happens
inside `VectorIndex.add_chunks` (hash-gated, per WP), so this stage's
"index" write is also where embedding is triggered.

Parameterized over the three index ports (not concrete adapters) so it
composes over whichever `VectorIndex`/`KeywordIndex`/`GraphStore`
instances the caller holds -- mirrors `RetrieveStage`'s injection
pattern. Side-effecting (writes to the indices) rather than a pure
data transform, same as `RetrieveStage`'s reads.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Protocol

from searchkernel.domain import Chunk
from searchkernel.pipeline.stage import SearchContext

_CHUNKS_KEY = "chunks"
_INDEXED_CHUNK_IDS_KEY = "indexed_chunk_ids"


class _VectorSink(Protocol):
    def add_chunks(self, chunks: list[Chunk]) -> None: ...


class _KeywordSink(Protocol):
    def add_chunks(self, chunks: list[Chunk]) -> None: ...


class _GraphSink(Protocol):
    def add_node(self, doc_id: str, metadata: dict) -> None: ...


class IndexStage:
    """Write `context.metadata["chunks"]` into vector, keyword, and graph.

    Expects `context.metadata["chunks"]` (`list[Chunk]`). Writes
    `context.metadata["indexed_chunk_ids"]` (`list[str]`, the chunk ids
    written, same order) for observability.
    """

    name = "index"

    def __init__(self, vector: _VectorSink, keyword: _KeywordSink, graph: _GraphSink):
        self._vector = vector
        self._keyword = keyword
        self._graph = graph

    def run(self, context: SearchContext) -> SearchContext:
        chunks: list[Chunk] = context.metadata[_CHUNKS_KEY]

        self._vector.add_chunks(chunks)
        self._keyword.add_chunks(chunks)
        for chunk in chunks:
            self._graph.add_node(chunk.chunk_id, chunk.metadata)

        metadata = dict(context.metadata)
        metadata[_INDEXED_CHUNK_IDS_KEY] = [chunk.chunk_id for chunk in chunks]
        return replace(context, metadata=metadata)
