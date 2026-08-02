"""Application adapter for the canonical searchkernel record pipeline."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from searchkernel.api import (
    CompressionStats,
    Record,
    SearchResultProvenance,
    SearchStrategyStats,
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
        self._pipeline = manager.kernel.pipeline
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
        **_: object,
    ) -> tuple[list[ChunkResult], CompressionStats, SearchStrategyStats]:
        del pipeline_config, project_context
        filters: dict[str, object] = {}
        if source_filter:
            filters["source_kinds"] = list(source_filter)
        outcome = await self.search(query, limit=max(top_k, top_n), filters=filters)
        filtered_results = [
            result
            for result in outcome.results
            if (
                not project_filter
                or result.record.metadata.get("project_id") in project_filter
            )
            and not self._is_excluded(result.record.metadata, excluded_files)
        ]
        selected_results = filtered_results[:top_n]
        maximum_score = max(
            (result.score for result in selected_results),
            default=0.0,
        )
        results = [
            self._to_chunk_result(
                result.record,
                result.score / maximum_score if maximum_score else result.score,
                result.provenance,
            )
            for result in selected_results
        ]
        self.last_query_execution_stats = {
            "degraded": outcome.degraded,
            "failures": [failure.message for failure in outcome.failures],
        }
        count = len(results)
        return (
            results,
            CompressionStats(count, count, count, count, count, count, 0),
            SearchStrategyStats(keyword_count=count),
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
        )

    async def drain_reindex(self) -> None:
        return None

    @staticmethod
    def _to_chunk_result(
        record: Record,
        score: float,
        provenance: SearchResultProvenance,
    ) -> ChunkResult:
        metadata = dict(record.metadata)
        metadata.setdefault("source_kind", record.source_kind)
        metadata.setdefault("source_id", record.source_id)
        return ChunkResult(
            chunk_id=str(metadata.get("chunk_id", record.source_id)),
            doc_id=str(metadata.get("doc_id", record.source_id)),
            score=score,
            header_path=str(metadata.get("header_path", "")),
            file_path=str(metadata.get("file_path", "")),
            project_id=metadata.get("project_id"),
            content=record.body,
            parent_chunk_id=metadata.get("parent_chunk_id"),
            provenance=provenance,
            metadata=metadata,
        )

    def _is_excluded(
        self,
        metadata: dict[str, object],
        excluded_files: set[str] | None,
    ) -> bool:
        if not excluded_files:
            return False
        file_path = str(metadata.get("file_path", ""))
        path = Path(file_path)
        candidates = {
            file_path,
            path.name,
            path.stem,
            str(path.with_suffix("")),
        }
        try:
            relative = path.resolve().relative_to(self.documents_path)
            candidates.add(str(relative))
            candidates.add(str(relative.with_suffix("")))
        except ValueError:
            pass
        return bool(candidates & excluded_files)


__all__ = ["CanonicalSearchAdapter"]
