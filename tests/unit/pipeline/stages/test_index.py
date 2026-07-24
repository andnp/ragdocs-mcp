from datetime import datetime, timezone

from searchkernel.models import Chunk
from searchkernel.pipeline.stage import SearchContext
from searchkernel.pipeline.stages.index import IndexStage


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
    return Chunk(
        chunk_id=chunk_id,
        doc_id="doc-1",
        content="hello world",
        metadata={"tag": "x"},
        chunk_index=0,
        header_path="",
        start_pos=0,
        end_pos=11,
        file_path="doc-1.md",
        modified_time=datetime.now(timezone.utc),
    )


def test_index_stage_writes_chunks_to_all_three_indices():
    vector, keyword, graph = _StubVector(), _StubKeyword(), _StubGraph()
    chunks = [_chunk("doc-1_chunk_0"), _chunk("doc-1_chunk_1")]

    result = IndexStage(vector, keyword, graph).run(
        SearchContext(query="", metadata={"chunks": chunks})
    )

    assert vector.added == chunks
    assert keyword.added == chunks
    assert graph.nodes == [
        ("doc-1_chunk_0", {"tag": "x"}),
        ("doc-1_chunk_1", {"tag": "x"}),
    ]
    assert result.metadata["indexed_chunk_ids"] == ["doc-1_chunk_0", "doc-1_chunk_1"]


def test_index_stage_does_not_mutate_input_context():
    context = SearchContext(query="", metadata={"chunks": [_chunk("doc-1_chunk_0")]})

    IndexStage(_StubVector(), _StubKeyword(), _StubGraph()).run(context)

    assert "indexed_chunk_ids" not in context.metadata
