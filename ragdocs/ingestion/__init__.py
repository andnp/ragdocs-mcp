"""Source-agnostic ingestion primitives."""

from searchkernel.ingestion.embedding import (
    EmbeddingBatchResult,
    EmbeddingInput,
    embed_and_upsert,
    embed_in_batches,
)

__all__ = ["EmbeddingBatchResult", "EmbeddingInput", "embed_and_upsert", "embed_in_batches"]
