from datetime import datetime, timezone

from searchkernel.chunking.base import ChunkingStrategy
from searchkernel.models import Chunk, Document
from searchkernel.pipeline.stage import SearchContext
from searchkernel.pipeline.stages.chunk import ChunkStage


class _StubChunker(ChunkingStrategy):
    def __init__(self, chunks: list[Chunk]):
        self._chunks = chunks

    def chunk_document(self, document) -> list[Chunk]:
        self.seen_document = document
        return self._chunks


def _document() -> Document:
    return Document(
        id="doc-1",
        content="hello world",
        metadata={},
        links=[],
        tags=[],
        file_path="doc-1.md",
        modified_time=datetime.now(timezone.utc),
    )


def _chunk() -> Chunk:
    return Chunk(
        chunk_id="doc-1_chunk_0",
        doc_id="doc-1",
        content="hello world",
        metadata={},
        chunk_index=0,
        header_path="",
        start_pos=0,
        end_pos=11,
        file_path="doc-1.md",
        modified_time=datetime.now(timezone.utc),
    )


def test_chunk_stage_delegates_to_chunker():
    document = _document()
    chunks = [_chunk()]
    chunker = _StubChunker(chunks)

    result = ChunkStage(chunker).run(
        SearchContext(query="", metadata={"document": document})
    )

    assert chunker.seen_document is document
    assert result.metadata["chunks"] == chunks


def test_chunk_stage_does_not_mutate_input_context():
    context = SearchContext(query="", metadata={"document": _document()})

    ChunkStage(_StubChunker([_chunk()])).run(context)

    assert "chunks" not in context.metadata
