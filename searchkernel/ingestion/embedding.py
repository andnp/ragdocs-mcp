"""Batch embedding mechanics shared by source-owned ingestion pipelines."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import islice

from searchkernel.domain import Vector
from searchkernel.ports.embedding import EmbeddingBatchProvider, EmbeddingSink


@dataclass(frozen=True, slots=True)
class EmbeddingInput:
    """Source data needed to generate and persist one embedding."""

    source_kind: str
    source_id: str
    text: str
    workspace_id: str | None = None
    source_updated_at: str | None = None


@dataclass(frozen=True, slots=True)
class EmbeddingBatchResult:
    """Counts from one complete embedding/upsert operation."""

    attempted: int
    stored: int
    rejected: int
    batches: int


def embed_and_upsert(
    inputs: list[EmbeddingInput],
    *,
    provider: EmbeddingBatchProvider,
    sink: EmbeddingSink,
    batch_size: int,
) -> EmbeddingBatchResult:
    """Embed inputs in bounded batches and persist each result.

    The complete provider response is validated before any writes from a batch
    occur, preventing silent truncation when an adapter returns the wrong count.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    if not inputs:
        return EmbeddingBatchResult(attempted=0, stored=0, rejected=0, batches=0)

    vectors = embed_in_batches(
        [item.text for item in inputs],
        provider=provider,
        batch_size=batch_size,
    )
    stored = 0
    rejected = 0
    for item, embedding in zip(inputs, vectors, strict=True):
        accepted = sink.upsert(
            source_kind=item.source_kind,
            source_id=item.source_id,
            workspace_id=item.workspace_id,
            model_name=provider.model_name,
            embedding=list(embedding),
            source_updated_at=item.source_updated_at,
        )
        if accepted is False:
            rejected += 1
        else:
            stored += 1

    return EmbeddingBatchResult(
        attempted=len(inputs),
        stored=stored,
        rejected=rejected,
        batches=(len(inputs) + batch_size - 1) // batch_size,
    )


def embed_in_batches(
    texts: list[str],
    *,
    provider: EmbeddingBatchProvider,
    batch_size: int,
) -> list[Vector]:
    """Embed texts in bounded batches and validate positional completeness."""
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    if not texts:
        return []

    vectors: list[Vector] = []
    iterator = iter(texts)
    while batch := list(islice(iterator, batch_size)):
        batch_vectors = provider.embed(batch)
        if len(batch_vectors) != len(batch):
            raise ValueError(
                f"Embedding provider {provider.model_name!r} returned {len(batch_vectors)} "
                f"vectors for {len(batch)} inputs"
            )
        vectors.extend(list(vector) for vector in batch_vectors)
    return vectors
