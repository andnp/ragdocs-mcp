"""Store ports: unified storage backends for vectors, keywords, graphs, and cache.

These ports unify the various storage needs of the kernel. Default implementation
uses Postgres + pgvector; legacy FAISS/SQLite adapters are kept as fallbacks.
"""

from typing import Any, Protocol, runtime_checkable

from searchkernel.domain import Record, Vector


@runtime_checkable
class VectorStore(Protocol):
    """Stores embeddings and supports ANN (approximate nearest neighbor) search."""

    def upsert(
        self, records: list[Record], model_name: str, dim: int
    ) -> None:
        """
        Upsert records with their embeddings into the store.

        Args:
            records: Records with embeddings populated (Record.embedding is set).
            model_name: Name of the embedding model used.
            dim: Dimensionality of the embeddings.

        Per-model embedding isolation ensures safe migration between models.
        """
        ...

    def search(
        self, query_vector: Vector, k: int, filters: dict[str, Any] | None = None
    ) -> list[tuple[str, float]]:
        """
        Search for the k nearest neighbors to a query vector.

        Args:
            query_vector: Embedding vector (must match stored dim).
            k: Number of results to return.
            filters: Optional filters (source-specific, opaque to core).

        Returns:
            List of (record_id, similarity_score) tuples, sorted descending.
        """
        ...

    def delete(self, record_ids: list[str]) -> None:
        """
        Delete records by ID.

        Args:
            record_ids: IDs to delete.
        """
        ...

    def epoch(self) -> int:
        """
        Return the current index epoch (version counter).

        Used to invalidate cache entries. Incremented on reindex/upsert.
        """
        ...


@runtime_checkable
class KeywordStore(Protocol):
    """Indexes and searches records by keyword/BM25."""

    def index(self, records: list[Record]) -> None:
        """
        Index records for full-text search.

        Args:
            records: Records to index (body and title are typically indexed).
        """
        ...

    def search(
        self, query: str, k: int, filters: dict[str, Any] | None = None
    ) -> list[tuple[str, float]]:
        """
        Search for top-k records matching the query.

        Args:
            query: Free-text search query.
            k: Number of results to return.
            filters: Optional filters (source-specific).

        Returns:
            List of (record_id, relevance_score) tuples, sorted descending.
        """
        ...


@runtime_checkable
class GraphStore(Protocol):
    """Stores and navigates graph relationships between records."""

    def upsert_edges(
        self,
        edges: list[tuple[str, str, str, float]],
    ) -> None:
        """
        Upsert edges in the graph.

        Args:
            edges: List of (source_id, target_id, edge_type, weight) tuples.
                   Updates if source+target+edge_type exists; inserts otherwise.
        """
        ...

    def neighbors(
        self,
        record_id: str,
        edge_types: list[str] | None = None,
        depth: int = 1,
    ) -> list[tuple[str, str, float]]:
        """
        Retrieve neighbors of a record in the graph.

        Args:
            record_id: Starting record ID.
            edge_types: Optional filter by edge type names.
            depth: Number of hops to traverse (default 1 for one-hop).

        Returns:
            List of (neighbor_id, edge_type, cumulative_weight) tuples.
        """
        ...


@runtime_checkable
class CacheStore(Protocol):
    """Caches stage outputs with epoch-based invalidation."""

    def get(self, key: str) -> Any | None:
        """
        Retrieve a cached value.

        Args:
            key: Cache key.

        Returns:
            Cached value, or None if not found or stale.
        """
        ...

    def set(self, key: str, value: Any, epoch: int) -> None:
        """
        Store a value with an associated epoch.

        Args:
            key: Cache key.
            value: Value to cache.
            epoch: Index epoch at cache time. Entries from older epochs
                   are considered stale.
        """
        ...

    def invalidate_epoch(self, epoch: int) -> None:
        """
        Invalidate all entries from an epoch or earlier.

        Args:
            epoch: Entries with epoch <= this are discarded.
        """
        ...
