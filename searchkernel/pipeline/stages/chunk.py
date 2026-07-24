"""ChunkStage: document-chunking ingestion stage.

Lifted from `IndexManager`'s three `self._chunker.chunk_document(document)`
call sites (`index_document`, `index_record`, `reconcile_indices`'s move
detection), the second phase of the ingestion path (discover -> chunk ->
embed -> index -> dedup/canonicalize -> re-embed/repair). Pure delegate to
a `ChunkingStrategy` instance -- same input, same output -- parameterized
over the strategy (like `RetrieveStage` over its searchers) since the
concrete chunker varies with `Config.chunking`.
"""

from __future__ import annotations

from dataclasses import replace

from searchkernel.chunking.base import ChunkingStrategy
from searchkernel.pipeline.stage import SearchContext

_DOCUMENT_KEY = "document"
_CHUNKS_KEY = "chunks"


class ChunkStage:
    """Chunk a document with a configured `ChunkingStrategy`.

    Expects `context.metadata["document"]`. Writes the resulting
    `list[Chunk]` to `context.metadata["chunks"]`.
    """

    name = "chunk"

    def __init__(self, chunker: ChunkingStrategy):
        self._chunker = chunker

    def run(self, context: SearchContext) -> SearchContext:
        document = context.metadata[_DOCUMENT_KEY]
        chunks = self._chunker.chunk_document(document)

        metadata = dict(context.metadata)
        metadata[_CHUNKS_KEY] = chunks
        return replace(context, metadata=metadata)
