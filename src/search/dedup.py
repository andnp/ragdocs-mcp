import hashlib
from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

from src.utils.similarity import cosine_similarity


def get_ngrams(text: str, n: int = 3) -> set[str]:
    text = text.lower().strip()
    if len(text) < n:
        return {text} if text else set()
    return {text[i:i + n] for i in range(len(text) - n + 1)}


def jaccard_similarity(text_a: str, text_b: str, n: int = 3) -> float:
    ngrams_a = get_ngrams(text_a, n)
    ngrams_b = get_ngrams(text_b, n)
    if not ngrams_a or not ngrams_b:
        return 0.0
    intersection = len(ngrams_a & ngrams_b)
    union = len(ngrams_a | ngrams_b)
    return intersection / union if union > 0 else 0.0


def deduplicate_by_ngram(
    results: list[tuple[str, float]],
    get_content: Callable[[str], str | None],
    threshold: float = 0.7,
    n: int = 3,
) -> tuple[list[tuple[str, float]], int]:
    if len(results) <= 1:
        return results, 0

    content_cache: dict[str, str] = {}
    ngram_cache: dict[str, set[str]] = {}

    for chunk_id, _ in results:
        content = get_content(chunk_id)
        if content is not None:
            content_cache[chunk_id] = content
            ngram_cache[chunk_id] = get_ngrams(content, n)

    kept: list[tuple[str, float]] = []
    removed = 0

    for chunk_id, score in results:
        if chunk_id not in ngram_cache:
            kept.append((chunk_id, score))
            continue

        is_duplicate = False
        current_ngrams = ngram_cache[chunk_id]

        for kept_id, _ in kept:
            if kept_id not in ngram_cache:
                continue
            kept_ngrams = ngram_cache[kept_id]
            if not current_ngrams or not kept_ngrams:
                continue
            intersection = len(current_ngrams & kept_ngrams)
            union = len(current_ngrams | kept_ngrams)
            sim = intersection / union if union > 0 else 0.0
            if sim >= threshold:
                is_duplicate = True
                removed += 1
                break

        if not is_duplicate:
            kept.append((chunk_id, score))

    return kept, removed


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
