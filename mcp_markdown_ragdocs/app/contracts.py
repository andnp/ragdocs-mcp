"""Typed application capabilities for the canonical search path."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol

from searchkernel.api import RecordSearchOutcome


class SearchExecutionPort(Protocol):
    """Execute a canonical search without exposing its implementation."""

    async def async_search(
        self,
        query: str,
        *,
        limit: int,
        filters: Mapping[str, object],
    ) -> RecordSearchOutcome: ...


class SearchDiagnosticsPort(Protocol):
    """Build transport-neutral diagnostics from a canonical outcome."""

    def __call__(self, outcome: RecordSearchOutcome) -> dict[str, object]: ...


class Reranker(Protocol):
    """Score application documents for optional result reranking."""

    model_name: str

    def rerank(self, query: str, documents: list[str]) -> list[float]: ...


__all__ = ["Reranker", "SearchDiagnosticsPort", "SearchExecutionPort"]
