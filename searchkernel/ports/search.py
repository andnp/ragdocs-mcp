"""SearchAPI port: the primary driving port for executing searches.

This is the main entry point for running a unified search across all sources.
"""

from typing import Any, Protocol, runtime_checkable

from searchkernel.domain import SearchResult


@runtime_checkable
class SearchAPI(Protocol):
    """Unified search endpoint spanning the kernel index and federated sources.

    This is the primary driving interface of the kernel. It orchestrates:
    - Fanning out to registered ContentSources (ingestible)
    - Querying the kernel's unified index (VectorStore + KeywordStore + GraphStore)
    - Federating to registered SearchableSources (federated)
    - Fusing and ranking results with RRF (Reciprocal Rank Fusion)
    - Optional reranking (retrieve-then-rerank-once)
    """

    async def search_anything(
        self,
        query: str,
        *,
        sources: list[str] | None = None,
        filters: dict[str, Any] | None = None,
        k: int = 10,
    ) -> list[SearchResult]:
        """
        Execute a unified search across kernel and federated sources.

        Args:
            query: The search query string.
            sources: Optional list of source_kind names to search.
                     If None, searches all registered sources.
            filters: Optional source-specific filters (opaque to core).
            k: Maximum number of results to return.

        Returns:
            Ranked list of SearchResults. Scores are normalized across sources
            via RRF fusion (and optional reranking). Results are in descending
            score order.
        """
        ...
