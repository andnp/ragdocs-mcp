from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

from src.utils.similarity import cosine_similarity


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
            relevance_scores[chunk_id] = cosine_similarity(embeddings[chunk_id], query_vec)
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
                        sim = cosine_similarity(chunk_emb, embeddings[sel_id])
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
