"""Source-agnostic ingestion primitives."""

from mcp_markdown_ragdocs.ingestion.embedding import (
    EmbeddingBatchResult,
    EmbeddingInput,
    embed_and_upsert,
    embed_in_batches,
)

__all__ = ["EmbeddingBatchResult", "EmbeddingInput", "embed_and_upsert", "embed_in_batches"]
