from dataclasses import dataclass
from typing import Protocol

import numpy as np

from src.models import ChunkResult


class EmbeddingModel(Protocol):
    def get_text_embedding(self, text: str) -> list[float]: ...


@dataclass
class DeduplicationResult:
    results: list[ChunkResult]
    original_count: int
    clusters_merged: int


def compute_similarity_matrix(embeddings: np.ndarray):
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    normalized = embeddings / norms
    return np.dot(normalized, normalized.T)


def cluster_by_similarity(
    similarity_matrix: np.ndarray,
    threshold: float = 0.85,
):
    n = similarity_matrix.shape[0]
    clusters: list[set[int]] = []
    assigned: set[int] = set()

    for i in range(n):
        if i in assigned:
            continue

        cluster = {i}
        assigned.add(i)

        for j in range(i + 1, n):
            if j in assigned:
                continue

            for member in cluster:
                if similarity_matrix[member, j] >= threshold:
                    cluster.add(j)
                    assigned.add(j)
                    break

        clusters.append(cluster)

    return clusters


def select_representatives(
    results: list[ChunkResult],
    clusters: list[set[int]],
):
    representatives = []

    for cluster in clusters:
        best_idx = max(cluster, key=lambda i: results[i].score)
        representatives.append(results[best_idx])

    representatives.sort(key=lambda r: r.score, reverse=True)
    return representatives


def get_embeddings_for_chunks(
    results: list[ChunkResult],
    embedding_model: EmbeddingModel,
):
    contents = [r.content for r in results]
    embeddings = [embedding_model.get_text_embedding(c) for c in contents]
    return np.array(embeddings)


def deduplicate_results(
    results: list[ChunkResult],
    embeddings: np.ndarray,
    similarity_threshold: float = 0.85,
):
    if len(results) <= 1:
        return DeduplicationResult(
            results=results,
            original_count=len(results),
            clusters_merged=0,
        )

    similarity_matrix = compute_similarity_matrix(embeddings)
    clusters = cluster_by_similarity(similarity_matrix, similarity_threshold)
    deduplicated = select_representatives(results, clusters)

    return DeduplicationResult(
        results=deduplicated,
        original_count=len(results),
        clusters_merged=len(results) - len(deduplicated),
    )
