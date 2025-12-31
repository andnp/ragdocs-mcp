import hashlib
from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray


def deduplicate_by_content_hash(
    results: list[tuple[str, float]],
    get_content: Callable[[str], str | None],
) -> tuple[list[tuple[str, float]], int]:
    seen_hashes: set[str] = set()
    kept: list[tuple[str, float]] = []
    removed = 0

    for chunk_id, score in results:
        content = get_content(chunk_id)
        if content is None:
            kept.append((chunk_id, score))
            continue

        content_hash = hashlib.md5(content.strip().encode()).hexdigest()
        if content_hash in seen_hashes:
            removed += 1
        else:
            seen_hashes.add(content_hash)
            kept.append((chunk_id, score))

    return kept, removed


def cosine_similarity(a: NDArray[np.floating], b: NDArray[np.floating]) -> float:
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def deduplicate_by_similarity(
    results: list[tuple[str, float]],
    get_embedding: Callable[[str], list[float] | None],
    similarity_threshold: float = 0.85,
) -> tuple[list[tuple[str, float]], int]:
    if len(results) <= 1:
        return results, 0

    embeddings: dict[str, NDArray[np.floating]] = {}
    for chunk_id, _ in results:
        emb = get_embedding(chunk_id)
        if emb is not None:
            embeddings[chunk_id] = np.array(emb, dtype=np.float64)

    kept: list[tuple[str, float]] = []
    removed: set[str] = set()
    clusters_merged = 0

    for chunk_id, score in results:
        if chunk_id in removed:
            continue
        if chunk_id not in embeddings:
            kept.append((chunk_id, score))
            continue

        is_duplicate = False
        for kept_id, _ in kept:
            if kept_id not in embeddings:
                continue
            sim = cosine_similarity(embeddings[chunk_id], embeddings[kept_id])
            if sim >= similarity_threshold:
                is_duplicate = True
                clusters_merged += 1
                break

        if not is_duplicate:
            kept.append((chunk_id, score))
        else:
            removed.add(chunk_id)

    return kept, clusters_merged
