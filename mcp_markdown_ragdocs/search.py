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

from mcp_markdown_ragdocs.git.results import aggregate_commit_results
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
        del pipeline_config, project_context
        filters: dict[str, object] = {}
        if source_filter:
            filters["source_kinds"] = list(source_filter)
        if project_filter:
            filters["project_ids"] = list(project_filter)
        if excluded_files:
            filters["excluded_files"] = sorted(excluded_files)
        # These are part of the canonical query contract.  Searchkernel may
        # consume them in a backend/policy, while the checks below remain a
        # defense-in-depth boundary for backends that do not.
        if min_score is not None:
            filters["min_score"] = min_score
        if similarity_threshold is not None:
            filters["similarity_threshold"] = similarity_threshold
        filters["max_chunks_per_doc"] = max_chunks_per_doc
        if retrieval_mode is not None:
            filters["retrieval_mode"] = retrieval_mode
        limit = max(top_k, top_n)
        if max_chunks_per_doc > 0:
            limit = max(limit, top_n * max(4, max_chunks_per_doc * 2))
        if source_filter == ["git_commit"]:
            limit = max(limit, top_n * 4)
        outcome = await self.search(query, limit=limit, filters=filters)
        filtered_results = [
            result
            for result in outcome.results
            if (
                not project_filter
                or result.record.metadata.get("project_id") in project_filter
            )
            and (min_score is None or result.score >= min_score)
            and not self._is_excluded(result.record.metadata, excluded_files)
        ]
        if max_chunks_per_doc > 0:
            counts: dict[str, int] = {}
            unique_results = []
            for result in filtered_results:
                doc_id = str(
                    result.record.metadata.get("doc_id", result.record.source_id)
                )
                if counts.get(doc_id, 0) >= max_chunks_per_doc:
                    continue
                counts[doc_id] = counts.get(doc_id, 0) + 1
                unique_results.append(result)
            filtered_results = unique_results
        results = [
            self._to_chunk_result(result.record, result.score, result.provenance)
            for result in filtered_results
        ]
        if source_filter == ["git_commit"]:
            results = aggregate_commit_results(results)
        results = results[:top_n]
        self.last_query_execution_stats = {
            "degraded": outcome.degraded,
            "failures": [failure.message for failure in outcome.failures],
        }
        strategy_counts = {"semantic": 0, "keyword": 0, "graph": 0, "tag_expansion": 0}
        for result in results:
            for strategy in result.provenance.strategies if result.provenance else ():
                strategy_name = {"vector": "semantic", "expansion": "tag_expansion"}.get(
                    strategy, strategy
                )
                if strategy_name in strategy_counts:
                    strategy_counts[strategy_name] += 1
        count = len(results)
        return (
            results,
            CompressionStats(
                len(outcome.results),
                len(filtered_results),
                len(filtered_results),
                len(filtered_results),
                len(filtered_results),
                count,
                0,
            ),
            SearchStrategyStats(
                vector_count=strategy_counts["semantic"],
                keyword_count=strategy_counts["keyword"],
                graph_count=strategy_counts["graph"],
                tag_expansion_count=strategy_counts["tag_expansion"],
            ),
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
        if not file_path:
            return False
        path = Path(file_path)
        candidates = {
            file_path,
            path.name,
            path.stem,
            str(path.with_suffix("")),
        }
        resolved = path.resolve()
        for root in self.documents_roots:
            try:
                relative = resolved.relative_to(root.resolve())
                candidates.add(str(relative))
                candidates.add(str(relative.with_suffix("")))
            except ValueError:
                continue
        return bool(candidates & excluded_files)


__all__ = ["CanonicalSearchAdapter"]
