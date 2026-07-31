from datetime import UTC, datetime

from searchkernel.chunking.base import ChunkingStrategy
from searchkernel.domain import Chunk
from searchkernel.pipeline.stage import SearchContext
from searchkernel.pipeline.stages.chunk import ChunkStage

from ragdocs.models import Document


def _with_hash(chunk):
    """Finalize a freshly-built domain.Chunk (test helper).

    domain.Chunk, unlike the legacy models.Chunk, does not auto-compute
    content_hash in __post_init__, and its metadata dict must stay JSON
    serializable (it flows into index/docstore persistence), so a raw
    datetime `modified_time` is normalized to ISO text.
    """
    if not chunk.content_hash:
        chunk.content_hash = chunk.compute_content_hash()
    modified_time = chunk.metadata.get("modified_time")
    if hasattr(modified_time, "isoformat"):
        chunk.metadata["modified_time"] = modified_time.isoformat()
    return chunk



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
        modified_time=datetime.now(UTC),
    )


def _chunk() -> Chunk:
    return _with_hash(Chunk(chunk_id="doc-1_chunk_0", record_id="doc-1", content="hello world", metadata={ "header_path": "", "start_pos": 0, "end_pos": 11, "file_path": "doc-1.md", "modified_time": datetime.now(UTC)}, chunk_index=0))


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
