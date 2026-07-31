from datetime import UTC, datetime

from searchkernel.domain import Chunk
from searchkernel.pipeline.stage import SearchContext
from searchkernel.pipeline.stages.index import IndexStage


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



class _StubVector:
    def __init__(self):
        self.added: list[Chunk] = []

    def add_chunks(self, chunks):
        self.added.extend(chunks)


class _StubKeyword:
    def __init__(self):
        self.added: list[Chunk] = []

    def add_chunks(self, chunks):
        self.added.extend(chunks)


class _StubGraph:
    def __init__(self):
        self.nodes: list[tuple[str, dict]] = []

    def add_node(self, doc_id, metadata):
        self.nodes.append((doc_id, metadata))


def _chunk(chunk_id: str) -> Chunk:
    return _with_hash(Chunk(chunk_id=chunk_id, record_id="doc-1", content="hello world", metadata={"tag": "x", "header_path": "", "start_pos": 0, "end_pos": 11, "file_path": "doc-1.md", "modified_time": datetime.now(UTC)}, chunk_index=0))


def test_index_stage_writes_chunks_to_all_three_indices():
    vector, keyword, graph = _StubVector(), _StubKeyword(), _StubGraph()
    chunks = [_chunk("doc-1_chunk_0"), _chunk("doc-1_chunk_1")]

    result = IndexStage(vector, keyword, graph).run(
        SearchContext(query="", metadata={"chunks": chunks})
    )

    assert vector.added == chunks
    assert keyword.added == chunks
    assert graph.nodes == [(chunk.chunk_id, chunk.metadata) for chunk in chunks]
    assert result.metadata["indexed_chunk_ids"] == ["doc-1_chunk_0", "doc-1_chunk_1"]


def test_index_stage_does_not_mutate_input_context():
    context = SearchContext(query="", metadata={"chunks": [_chunk("doc-1_chunk_0")]})

    IndexStage(_StubVector(), _StubKeyword(), _StubGraph()).run(context)

    assert "indexed_chunk_ids" not in context.metadata
