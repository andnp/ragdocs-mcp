"""Embedding ports for providers and source-owned embedding sinks."""

from typing import Protocol, runtime_checkable

from searchkernel.domain import Vector


@runtime_checkable
class EmbeddingBatchProvider(Protocol):
    """Generates a batch of embeddings without imposing dimension policy."""

    model_name: str

    def embed(self, texts: list[str]) -> list[Vector]:
        """Return one embedding for each input text, in input order."""
        ...


@runtime_checkable
class EmbeddingSink(Protocol):
    """Persists one source embedding with source-owned write policy."""

    def upsert(
        self,
        *,
        source_kind: str,
        source_id: str,
        workspace_id: str | None,
        model_name: str,
        embedding: Vector,
        source_updated_at: str | None = None,
    ) -> bool:
        """Persist an embedding and report whether the write was accepted."""
        ...


@runtime_checkable
class EmbeddingProvider(EmbeddingBatchProvider, Protocol):
    """Embedding provider with an explicit, stable vector dimension."""

    dim: int
