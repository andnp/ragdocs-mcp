"""Reranker port: adapters for relevance scoring and reranking.

Pluggable reranking models for refining search results. Implementations can wrap
HuggingFace, Ollama, or other reranking services. Scores are normalized to [0, 1]
with higher values indicating greater relevance.
"""

from typing import Protocol, runtime_checkable


@runtime_checkable
class Reranker(Protocol):
    """Scores document relevance against a query.

    Attributes:
        model_name: Stable identifier for this reranking model
                    (e.g., "Qwen/Qwen3-Reranker-0.6B").
    """

    model_name: str

    def rerank(self, query: str, documents: list[str]) -> list[float]:
        """
        Score a list of documents for relevance to a query.

        Args:
            query: The search query string.
            documents: List of document texts to score.

        Returns:
            List of relevance scores, one per input document, in the same order.
            Each score is a float in [0, 1]; higher = more relevant.
        """
        ...
