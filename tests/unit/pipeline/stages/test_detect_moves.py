from datetime import UTC, datetime

from searchkernel.domain import Chunk
from searchkernel.pipeline.stage import SearchContext
from searchkernel.pipeline.stages.detect_moves import DetectMovesStage


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



class _StubHashStore:
    def __init__(self, chunks_by_document: dict[str, list[tuple[str, str]]]):
        self._chunks_by_document = chunks_by_document

    def get_chunks_by_document(self, doc_id: str):
        return self._chunks_by_document.get(doc_id)


def _chunk(chunk_id: str, doc_id: str, content_hash: str) -> Chunk:
    chunk = _with_hash(Chunk(chunk_id=chunk_id, record_id=doc_id, content="content", metadata={ "header_path": "", "start_pos": 0, "end_pos": 7, "file_path": f"{doc_id}.md", "modified_time": datetime.now(UTC)}, chunk_index=0))
    object.__setattr__(chunk, "content_hash", content_hash)
    return chunk


def test_detect_moves_stage_matches_identical_content():
    hash_store = _StubHashStore({"old_doc": [("old_doc_chunk_0", "hash_a")]})
    new_chunks = [_chunk("new_doc_chunk_0", "new_doc", "hash_a")]

    context = SearchContext(
        query="",
        metadata={
            "removed_doc_ids": {"old_doc"},
            "added_docs": {"new_doc": new_chunks},
            "move_detection_threshold": 0.8,
        },
    )

    result = DetectMovesStage(hash_store).run(context)

    assert result.metadata["moved_files"] == {"old_doc": "new_doc"}


def test_detect_moves_stage_skips_below_threshold():
    hash_store = _StubHashStore(
        {"old_doc": [("old_doc_chunk_0", "hash_a"), ("old_doc_chunk_1", "hash_b")]}
    )
    new_chunks = [_chunk("new_doc_chunk_0", "new_doc", "hash_a")]

    context = SearchContext(
        query="",
        metadata={
            "removed_doc_ids": {"old_doc"},
            "added_docs": {"new_doc": new_chunks},
            "move_detection_threshold": 0.8,
        },
    )

    result = DetectMovesStage(hash_store).run(context)

    assert result.metadata["moved_files"] == {}


def test_detect_moves_stage_does_not_mutate_input_context():
    context = SearchContext(
        query="",
        metadata={
            "removed_doc_ids": set(),
            "added_docs": {},
            "move_detection_threshold": 0.8,
        },
    )

    DetectMovesStage(_StubHashStore({})).run(context)

    assert "moved_files" not in context.metadata
