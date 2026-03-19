from datetime import datetime, timezone

from src.indices.vector import VectorIndex
from src.models import Chunk
from src.search.graph_expansion import build_graph_chunk_candidates


def _make_chunk(doc_id: str, chunk_index: int) -> Chunk:
    return Chunk(
        chunk_id=f"{doc_id}_chunk_{chunk_index}",
        doc_id=doc_id,
        content=f"Content for {doc_id} chunk {chunk_index}",
        metadata={},
        chunk_index=chunk_index,
        header_path=f"Section {chunk_index}",
        start_pos=0,
        end_pos=10,
        file_path=f"{doc_id}.md",
        modified_time=datetime.now(timezone.utc),
    )


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