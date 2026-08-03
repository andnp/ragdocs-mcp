"""Application adapter for the canonical searchkernel record pipeline."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from searchkernel.api import (
    CompressionStats,
    SearchStrategyStats,
)

from mcp_markdown_ragdocs.app.search import (
    ApplicationSearchUseCase,
    SearchQuery,
)
from mcp_markdown_ragdocs.models import ChunkResult


@dataclass(frozen=True)
class SearchPipelineConfig:
    """Compatibility configuration for callers still passing pipeline options."""

    min_confidence: float = 0.3
    dedup_threshold: float = 0.85
    reranking_enabled: bool = True


def filter_by_score(
    results: Sequence[ChunkResult],
    min_score: float = 0.3,
) -> list[ChunkResult]:
    return [result for result in results if result.score >= min_score]


class CanonicalSearchAdapter:
    """Preserve ragdocs' query tuple while using RecordSearchPipeline."""

    def __init__(self, manager: Any, *, documents_path: Path | None = None) -> None:
        config = getattr(manager, "_config", None)
        indexing = getattr(config, "indexing", None)
        self.documents_path = documents_path or Path(
            getattr(indexing, "documents_path", ".")
        )
        self.documents_roots = tuple(
            Path(root) for root in getattr(manager, "_documents_roots", (self.documents_path,))
        )
        self._pipeline = manager.kernel.pipeline
        self.search_use_case = ApplicationSearchUseCase(
            self._pipeline,
            documents_roots=self.documents_roots,
        )
        self.last_query_execution_stats: dict[str, object] = {}

    async def search(
        self,
        query: str,
        *,
        limit: int = 10,
        filters: dict[str, object] | None = None,
    ):
        return await self._pipeline.async_search(
            query,
            limit=limit,
            filters=dict(filters or {}),
        )

    async def query(
        self,
        query: str,
        *,
        top_k: int,
        top_n: int,
        pipeline_config: object | None = None,
        project_filter: Sequence[str] | None = None,
        source_filter: Sequence[str] | None = None,
        project_context: str | None = None,
        excluded_files: set[str] | None = None,
        min_score: float | None = None,
        similarity_threshold: float | None = None,
        max_chunks_per_doc: int = 0,
        retrieval_mode: str | None = None,
        **_: object,
    ) -> tuple[list[ChunkResult], CompressionStats, SearchStrategyStats]:
        del pipeline_config
        execution = await self.search_use_case.execute(
            SearchQuery(
                query=query,
                top_n=top_n,
                top_k=top_k,
                project_filter=tuple(project_filter or ()),
                source_filter=tuple(source_filter or ()),
                project_context=project_context,
                excluded_files=frozenset(excluded_files or ()),
                min_score=min_score,
                similarity_threshold=similarity_threshold,
                max_chunks_per_doc=max_chunks_per_doc,
                retrieval_mode=retrieval_mode,
            ),
            search=self.search,
        )
        self.last_query_execution_stats = execution.query_execution_stats
        return execution.results, execution.compression_stats, execution.strategy_stats

    def _is_excluded(
        self,
        metadata: dict[str, object],
        excluded_files: set[str] | None,
    ) -> bool:
        return self.search_use_case._is_excluded(
            metadata,
            frozenset(excluded_files or ()),
        )

    async def query_with_hypothesis(
        self,
        query: str,
        *,
        top_k: int,
        top_n: int,
        pipeline_config: object | None = None,
        project_filter: Sequence[str] | None = None,
        source_filter: Sequence[str] | None = None,
        project_context: str | None = None,
        excluded_files: set[str] | None = None,
    ):
        return await self.query(
            query,
            top_k=top_k,
            top_n=top_n,
            pipeline_config=pipeline_config,
            project_filter=project_filter,
            source_filter=source_filter,
            project_context=project_context,
            excluded_files=excluded_files,
            retrieval_mode="semantic_only",
        )

    async def drain_reindex(self) -> None:
        return None

__all__ = ["CanonicalSearchAdapter", "SearchPipelineConfig", "filter_by_score"]
