"""Thread-safe, bounded cache for repeated query embeddings."""

from __future__ import annotations

import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field
from threading import Event, Lock

from searchkernel.domain import Vector


@dataclass(slots=True)
class _CachedEmbedding:
    embedding: Vector
    expires_at: float


@dataclass(slots=True)
class _InFlightEmbedding:
    completed: Event = field(default_factory=Event)
    embedding: Vector | None = None
    error: Exception | None = None


class QueryEmbeddingCache:
    """Caches exact ``(model_name, query)`` results and coalesces concurrent misses."""

    def __init__(
        self,
        *,
        ttl_seconds: float = 300.0,
        max_entries: int = 128,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be > 0")
        if max_entries < 1:
            raise ValueError("max_entries must be >= 1")
        self._ttl_seconds = ttl_seconds
        self._max_entries = max_entries
        self._clock = clock
        self._cache: OrderedDict[tuple[str, str], _CachedEmbedding] = OrderedDict()
        self._inflight: dict[tuple[str, str], _InFlightEmbedding] = {}
        self._lock = Lock()

    def get_or_compute(
        self,
        *,
        model_name: str,
        query: str,
        compute: Callable[[], Vector],
    ) -> Vector:
        key = (model_name, query)
        cached = self._load(key)
        if cached is not None:
            return cached

        with self._lock:
            inflight = self._inflight.get(key)
            if inflight is None:
                inflight = _InFlightEmbedding()
                self._inflight[key] = inflight
                is_leader = True
            else:
                is_leader = False

        if not is_leader:
            inflight.completed.wait()
            if inflight.embedding is not None:
                return list(inflight.embedding)
            if inflight.error is not None:
                raise inflight.error
            raise RuntimeError("Query embedding coalescing completed without a result")

        try:
            embedding = list(compute())
            inflight.embedding = embedding
            self._store(key, embedding)
            return list(embedding)
        except Exception as exc:
            inflight.error = exc
            raise
        finally:
            inflight.completed.set()
            with self._lock:
                if self._inflight.get(key) is inflight:
                    self._inflight.pop(key, None)

    def clear(self) -> None:
        """Clear completed and in-flight entries."""
        with self._lock:
            self._cache.clear()
            self._inflight.clear()

    def _load(self, key: tuple[str, str]) -> Vector | None:
        now = self._clock()
        with self._lock:
            self._evict_expired(now)
            cached = self._cache.get(key)
            if cached is None:
                return None
            self._cache.move_to_end(key)
            return list(cached.embedding)

    def _store(self, key: tuple[str, str], embedding: Vector) -> None:
        now = self._clock()
        with self._lock:
            self._evict_expired(now)
            self._cache[key] = _CachedEmbedding(
                embedding=list(embedding),
                expires_at=now + self._ttl_seconds,
            )
            self._cache.move_to_end(key)
            while len(self._cache) > self._max_entries:
                self._cache.popitem(last=False)

    def _evict_expired(self, now: float) -> None:
        expired = [
            key
            for key, cached in self._cache.items()
            if cached.expires_at <= now
        ]
        for key in expired:
            self._cache.pop(key, None)


_DEFAULT_QUERY_EMBEDDING_CACHE = QueryEmbeddingCache()


def get_or_compute_query_embedding(
    *,
    model_name: str,
    query: str,
    compute: Callable[[], Vector],
) -> Vector:
    """Use the process-wide query embedding cache."""
    return _DEFAULT_QUERY_EMBEDDING_CACHE.get_or_compute(
        model_name=model_name,
        query=query,
        compute=compute,
    )


def clear_query_embedding_cache() -> None:
    """Clear the process-wide query embedding cache."""
    _DEFAULT_QUERY_EMBEDDING_CACHE.clear()
