from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.indices.keyword import KeywordIndex
    from src.indices.vector import VectorIndex
    from src.models import ChunkResult
    from src.search.chunk_hydrator import ChunkHydrator


@dataclass
class QueryExecutionStats:
    metadata_lookups: int = 0
    metadata_cache_hits: int = 0
    content_lookups: int = 0
    content_cache_hits: int = 0
    embedding_fetches: int = 0
    embedding_cache_hits: int = 0
    parent_lookups: int = 0
    parent_cache_hits: int = 0

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


class QueryExecutionContext:
    def __init__(
        self,
        vector: VectorIndex,
        keyword: KeywordIndex,
        chunk_hydrator: ChunkHydrator,
    ) -> None:
        self._vector = vector
        self._keyword = keyword
        self._chunk_hydrator = chunk_hydrator
        self._vector_chunk_cache: dict[str, dict[str, Any] | None] = {}
        self._keyword_chunk_cache: dict[str, dict[str, Any] | None] = {}
        self._content_cache: dict[str, str | None] = {}
        self._embedding_cache: dict[str, list[float] | None] = {}
        self._parent_content_cache: dict[str, str | None] = {}
        self._stats = QueryExecutionStats()

    @property
    def stats(self) -> QueryExecutionStats:
        return self._stats

    def get_vector_chunk(self, chunk_id: str) -> dict[str, Any] | None:
        if chunk_id in self._vector_chunk_cache:
            self._stats.metadata_cache_hits += 1
            return self._vector_chunk_cache[chunk_id]

        self._stats.metadata_lookups += 1
        chunk_data = self._vector.get_chunk_by_id(chunk_id)
        self._vector_chunk_cache[chunk_id] = chunk_data
        return chunk_data

    def get_keyword_chunk(self, chunk_id: str) -> dict[str, Any] | None:
        if chunk_id in self._keyword_chunk_cache:
            self._stats.metadata_cache_hits += 1
            return self._keyword_chunk_cache[chunk_id]

        self._stats.metadata_lookups += 1
        chunk_data = self._keyword.get_chunk_by_id(chunk_id)
        self._keyword_chunk_cache[chunk_id] = chunk_data
        return chunk_data

    def get_chunk_embedding(self, chunk_id: str) -> list[float] | None:
        if chunk_id in self._embedding_cache:
            self._stats.embedding_cache_hits += 1
            return self._embedding_cache[chunk_id]

        self._stats.embedding_fetches += 1
        embedding = self._vector.get_embedding_for_chunk(chunk_id)
        self._embedding_cache[chunk_id] = embedding
        return embedding

    def get_parent_chunk(self, parent_chunk_id: str) -> dict[str, Any] | None:
        self._stats.parent_lookups += 1
        if parent_chunk_id in self._vector_chunk_cache:
            self._stats.parent_cache_hits += 1
        return self.get_vector_chunk(parent_chunk_id)

    def get_parent_content(self, parent_chunk_id: str) -> str | None:
        self._stats.parent_lookups += 1
        if parent_chunk_id in self._parent_content_cache:
            self._stats.parent_cache_hits += 1
            return self._parent_content_cache[parent_chunk_id]

        parent_content = self._vector.get_parent_content(parent_chunk_id)
        self._parent_content_cache[parent_chunk_id] = parent_content
        return parent_content

    def get_chunk_content(self, chunk_id: str) -> str | None:
        if chunk_id in self._content_cache:
            self._stats.content_cache_hits += 1
            return self._content_cache[chunk_id]

        self._stats.content_lookups += 1
        content = self._chunk_hydrator.get_content(chunk_id, query_context=self)
        self._content_cache[chunk_id] = content
        return content

    def hydrate_chunk_result(self, chunk_id: str, score: float) -> ChunkResult | None:
        return self._chunk_hydrator.hydrate_chunk_result(
            chunk_id,
            score,
            query_context=self,
        )