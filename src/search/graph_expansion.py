from dataclasses import dataclass

from src.indices.vector import VectorIndex

_DEFAULT_MAX_CHUNKS_PER_DOC = 1


@dataclass(frozen=True)
class GraphCandidateExpansionPolicy:
    max_documents: int
    max_chunks_per_document: int = _DEFAULT_MAX_CHUNKS_PER_DOC


def build_graph_candidate_expansion_policy(
    top_k: int,
    max_chunks_per_document: int = _DEFAULT_MAX_CHUNKS_PER_DOC,
) -> GraphCandidateExpansionPolicy:
    bounded_top_k = max(1, top_k)
    bounded_chunks_per_document = max(1, max_chunks_per_document)
    return GraphCandidateExpansionPolicy(
        max_documents=bounded_top_k,
        max_chunks_per_document=bounded_chunks_per_document,
    )


def build_graph_chunk_candidates(
    neighbor_doc_ids: list[str],
    vector_index: VectorIndex,
    top_k: int,
    excluded_chunk_ids: set[str] | None = None,
    policy: GraphCandidateExpansionPolicy | None = None,
) -> list[str]:
    if not neighbor_doc_ids or top_k <= 0:
        return []

    expansion_policy = policy or build_graph_candidate_expansion_policy(top_k)
    seen_chunk_ids = set(excluded_chunk_ids or ())
    seen_doc_ids: set[str] = set()
    graph_chunk_ids: list[str] = []

    for doc_id in neighbor_doc_ids:
        if doc_id in seen_doc_ids:
            continue

        representative_chunk_ids = _select_representative_chunk_ids(
            doc_id,
            vector_index,
            seen_chunk_ids,
            expansion_policy.max_chunks_per_document,
        )
        if not representative_chunk_ids:
            continue

        seen_doc_ids.add(doc_id)
        graph_chunk_ids.extend(representative_chunk_ids)
        seen_chunk_ids.update(representative_chunk_ids)

        if len(seen_doc_ids) >= expansion_policy.max_documents:
            break

    return graph_chunk_ids


def _select_representative_chunk_ids(
    doc_id: str,
    vector_index: VectorIndex,
    excluded_chunk_ids: set[str],
    max_chunks_per_document: int,
) -> list[str]:
    selected_chunk_ids: list[str] = []

    for chunk_id in vector_index.get_chunk_ids_for_document(doc_id):
        if chunk_id in excluded_chunk_ids:
            continue

        selected_chunk_ids.append(chunk_id)
        if len(selected_chunk_ids) >= max_chunks_per_document:
            break

    return selected_chunk_ids