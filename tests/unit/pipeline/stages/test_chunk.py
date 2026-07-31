from datetime import UTC, datetime

from searchkernel.chunking.base import ChunkingStrategy
from searchkernel.domain import Chunk, Record
from searchkernel.pipeline.stage import SearchContext
from searchkernel.pipeline.stages.chunk import ChunkStage


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

    def chunk_record(self, record) -> list[Chunk]:
        self.seen_record = record
        return self._chunks


def _record() -> Record:
    now = datetime.now(UTC)
    return Record(
        source_kind="note",
        source_id="doc-1",
        title="doc-1",
        body="hello world",
        created_at=now,
        updated_at=now,
        metadata={},
        uri="doc-1.md",
    )


def _chunk() -> Chunk:
    return _with_hash(Chunk(chunk_id="doc-1_chunk_0", record_id="doc-1", content="hello world", metadata={ "header_path": "", "start_pos": 0, "end_pos": 11, "file_path": "doc-1.md", "modified_time": datetime.now(UTC)}, chunk_index=0))


def test_chunk_stage_delegates_to_chunker():
    record = _record()
    chunks = [_chunk()]
    chunker = _StubChunker(chunks)

    result = ChunkStage(chunker).run(
        SearchContext(query="", metadata={"record": record})
    )

    assert chunker.seen_record is record
    assert result.metadata["chunks"] == chunks


def test_chunk_stage_does_not_mutate_input_context():
    context = SearchContext(query="", metadata={"record": _record()})

    ChunkStage(_StubChunker([_chunk()])).run(context)

    assert "chunks" not in context.metadata
