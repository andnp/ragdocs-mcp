from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray


def _cosine_similarity(a: NDArray[np.floating], b: NDArray[np.floating]) -> float:
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def select_mmr(
    query_embedding: list[float],
    candidates: list[tuple[str, float]],
    get_embedding: Callable[[str], list[float] | None],
    lambda_param: float = 0.7,
    top_n: int = 10,
) -> list[tuple[str, float]]:
    if not candidates:
        return []
    if len(candidates) <= 1:
        return candidates[:top_n]

    query_vec = np.array(query_embedding, dtype=np.float64)

    embeddings: dict[str, NDArray[np.floating]] = {}
    relevance_scores: dict[str, float] = {}

    for chunk_id, score in candidates:
        emb = get_embedding(chunk_id)
        if emb is not None:
            embeddings[chunk_id] = np.array(emb, dtype=np.float64)
            relevance_scores[chunk_id] = _cosine_similarity(embeddings[chunk_id], query_vec)
        else:
            relevance_scores[chunk_id] = score

    selected: list[tuple[str, float]] = []
    remaining = {chunk_id for chunk_id, _ in candidates}

    while remaining and len(selected) < top_n:
        best_id = None
        best_mmr = float("-inf")

        for chunk_id in remaining:
            relevance = relevance_scores[chunk_id]

            max_sim_to_selected = 0.0
            if selected and chunk_id in embeddings:
                chunk_emb = embeddings[chunk_id]
                for sel_id, _ in selected:
                    if sel_id in embeddings:
                        sim = _cosine_similarity(chunk_emb, embeddings[sel_id])
                        max_sim_to_selected = max(max_sim_to_selected, sim)

            mmr_score = lambda_param * relevance - (1 - lambda_param) * max_sim_to_selected

            if mmr_score > best_mmr:
                best_mmr = mmr_score
                best_id = chunk_id

        if best_id is not None:
            original_score = next(
                (score for cid, score in candidates if cid == best_id), 0.0
            )
            selected.append((best_id, original_score))
            remaining.remove(best_id)
        else:
            break

    return selected
