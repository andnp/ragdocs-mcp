from datetime import UTC, datetime

from searchkernel.domain import Chunk
from searchkernel.pipeline.stage import SearchContext
from searchkernel.pipeline.stages.apply_move import ApplyMoveStage


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
    def __init__(self, ok: bool = True):
        self._ok = ok
        self.calls: list[tuple[str, str, dict]] = []

    def update_chunk_path(self, old_chunk_id, new_chunk_id, new_metadata):
        self.calls.append((old_chunk_id, new_chunk_id, new_metadata))
        return self._ok


class _StubKeyword:
    def __init__(self, ok: bool = True):
        self._ok = ok
        self.calls: list[tuple[str, Chunk]] = []

    def move_chunk(self, old_chunk_id, new_chunk):
        self.calls.append((old_chunk_id, new_chunk))
        return self._ok


class _StubGraph:
    def __init__(self):
        self.calls: list[tuple[str, str]] = []

    def rename_node(self, old_chunk_id, new_chunk_id):
        self.calls.append((old_chunk_id, new_chunk_id))
        return True


class _StubHashStore:
    def __init__(self, chunks_by_document: dict[str, list[tuple[str, str]]]):
        self._chunks_by_document = chunks_by_document
        self.removed: list[str] = []
        self.set_hashes: list[tuple[str, str]] = []

    def get_chunks_by_document(self, doc_id: str):
        return self._chunks_by_document.get(doc_id)

    def remove_document(self, doc_id: str) -> None:
        self.removed.append(doc_id)

    def set_hash(self, chunk_id: str, content_hash: str) -> None:
        self.set_hashes.append((chunk_id, content_hash))


def _chunk(chunk_id: str, doc_id: str, content_hash: str) -> Chunk:
    chunk = _with_hash(Chunk(chunk_id=chunk_id, record_id=doc_id, content="content", metadata={ "header_path": "", "start_pos": 0, "end_pos": 7, "file_path": f"{doc_id}.md", "modified_time": datetime.now(UTC)}, chunk_index=0))
    object.__setattr__(chunk, "content_hash", content_hash)
    return chunk


def _context(old_doc_id, new_doc_id, new_chunks) -> SearchContext:
    return SearchContext(
        query="",
        metadata={
            "old_doc_id": old_doc_id,
            "new_doc_id": new_doc_id,
            "new_chunks": new_chunks,
        },
    )


def test_apply_move_stage_returns_false_when_no_old_chunks():
    stage = ApplyMoveStage(
        _StubVector(), _StubKeyword(), _StubGraph(), _StubHashStore({}), 0.8
    )

    result = stage.run(_context("old_doc", "new_doc", [_chunk("c0", "new_doc", "h")]))

    assert result.metadata["move_applied"] is False
    assert result.metadata["hash_store_updated"] is False


def test_apply_move_stage_moves_matching_chunks():
    hash_store = _StubHashStore({"old_doc": [("old_c0", "h")]})
    vector = _StubVector()
    keyword = _StubKeyword()
    graph = _StubGraph()
    stage = ApplyMoveStage(vector, keyword, graph, hash_store, 0.8)
    new_chunks = [_chunk("new_c0", "new_doc", "h")]

    result = stage.run(_context("old_doc", "new_doc", new_chunks))

    assert result.metadata["move_applied"] is True
    assert result.metadata["moved_chunk_count"] == 1
    assert result.metadata["hash_store_updated"] is True
    assert vector.calls == [("old_c0", "new_c0", {
        "doc_id": "new_doc",
        "chunk_id": "new_c0",
        "file_path": "new_doc.md",
        "header_path": "",
        **new_chunks[0].metadata,
    })]
    assert keyword.calls == [("old_c0", new_chunks[0])]
    assert graph.calls == [("old_c0", "new_c0")]
    assert hash_store.removed == ["old_doc"]
    assert hash_store.set_hashes == [("new_c0", "h")]


def test_apply_move_stage_falls_back_when_too_many_chunks_fail():
    hash_store = _StubHashStore({"old_doc": [("old_c0", "h0"), ("old_c1", "h1")]})
    stage = ApplyMoveStage(_StubVector(), _StubKeyword(), _StubGraph(), hash_store, 0.9)
    new_chunks = [
        _chunk("new_c0", "new_doc", "h0"),
        _chunk("new_c1", "new_doc", "unmatched"),
    ]

    result = stage.run(_context("old_doc", "new_doc", new_chunks))

    assert result.metadata["move_applied"] is False
    assert result.metadata["moved_chunk_count"] == 1
    assert result.metadata["hash_store_updated"] is True


def test_apply_move_stage_does_not_mutate_input_context():
    context = _context("old_doc", "new_doc", [])

    ApplyMoveStage(
        _StubVector(), _StubKeyword(), _StubGraph(), _StubHashStore({}), 0.8
    ).run(context)

    assert "move_applied" not in context.metadata
