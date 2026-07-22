from __future__ import annotations

import copy
from collections import OrderedDict
from dataclasses import dataclass
from typing import Generic, TypeVar


@dataclass(frozen=True)
class QueryResultCacheKey:
    query_text: str
    top_k: int
    top_n: int
    min_confidence: float
    max_chunks_per_doc: int
    dedup_threshold: float
    reranking_enabled: bool
    rerank_top_n: int
    excluded_files: tuple[str, ...]
    project_filter: tuple[str, ...]
    project_context: str | None
    index_state_version: int


ValueT = TypeVar("ValueT")


class QueryResultCache(Generic[ValueT]):
    def __init__(self, max_entries: int = 64):
        self._max_entries = max(1, max_entries)
        self._entries: OrderedDict[QueryResultCacheKey, ValueT] = OrderedDict()

    def get(self, key: QueryResultCacheKey) -> ValueT | None:
        value = self._entries.get(key)
        if value is None:
            return None

        self._entries.move_to_end(key)
        return copy.deepcopy(value)

    def set(self, key: QueryResultCacheKey, value: ValueT) -> None:
        self._entries[key] = copy.deepcopy(value)
        self._entries.move_to_end(key)

        while len(self._entries) > self._max_entries:
            self._entries.popitem(last=False)