from datetime import UTC, datetime

from searchkernel.domain import Chunk
from searchkernel.indices.vector import VectorIndex
from searchkernel.search.graph_expansion import build_graph_chunk_candidates


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



def _make_chunk(doc_id: str, chunk_index: int) -> Chunk:
    return _with_hash(Chunk(chunk_id=f"{doc_id}_chunk_{chunk_index}", record_id=doc_id, content=f"Content for {doc_id} chunk {chunk_index}", metadata={ "header_path": f"Section {chunk_index}", "start_pos": 0, "end_pos": 10, "file_path": f"{doc_id}.md", "modified_time": datetime.now(UTC)}, chunk_index=chunk_index))


def test_build_graph_chunk_candidates_limits_results_per_doc_and_total(
    shared_embedding_model,
):
    vector = VectorIndex(embedding_model=shared_embedding_model)

    for doc_id in ["doc_a", "doc_b", "doc_c"]:
        for chunk_index in range(3):
            vector.add_chunk(_make_chunk(doc_id, chunk_index))

    graph_chunk_ids = build_graph_chunk_candidates(
        ["doc_a", "doc_b", "doc_c"],
        vector,
        top_k=2,
    )

    assert graph_chunk_ids == ["doc_a_chunk_0", "doc_b_chunk_0"]


def test_build_graph_chunk_candidates_skips_existing_direct_chunk_ids(
    shared_embedding_model,
):
    vector = VectorIndex(embedding_model=shared_embedding_model)

    for chunk_index in range(3):
        vector.add_chunk(_make_chunk("doc_a", chunk_index))

    graph_chunk_ids = build_graph_chunk_candidates(
        ["doc_a"],
        vector,
        top_k=1,
        excluded_chunk_ids={"doc_a_chunk_0"},
    )

    assert graph_chunk_ids == ["doc_a_chunk_1"]