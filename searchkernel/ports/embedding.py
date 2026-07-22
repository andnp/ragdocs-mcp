"""EmbeddingProvider port: adapters for generating embeddings.

Pluggable embedding models with a registry. Implementations can wrap
HuggingFace, Ollama, or other embedding services.
"""

from typing import Protocol, runtime_checkable

from searchkernel.domain import Vector


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Embeds text into vectors for semantic search.

    Attributes:
        model_name: Stable identifier for this embedding model
                    (e.g., "Qwen3-Embedding-0.6B", "all-MiniLM-L6-v2").
        dim: Embedding dimensionality (e.g., 1024, 768).
    """

    model_name: str
    dim: int

    def embed(self, texts: list[str]) -> list[Vector]:
        """
        Embed a batch of texts into vectors.

        Args:
            texts: List of texts (typically 1-256 items for batching).

        Returns:
            List of embedding vectors, one per input text, in the same order.
            Each vector is a list of floats with length == self.dim.
        """
        ...
