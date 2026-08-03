"""Application-owned search use case over the canonical record pipeline."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from importlib import import_module
from pathlib import Path
from typing import Any, Protocol

from searchkernel.api import (
    CompressionStats,
    Record,
    RecordSearchConfig,
    RecordSearchOutcome,
    SearchResultProvenance,
    SearchStrategyStats,
)

from mcp_markdown_ragdocs.config import SearchConfig
from mcp_markdown_ragdocs.git.results import aggregate_commit_results
from mcp_markdown_ragdocs.models import ChunkResult


class SearchKernelBoundary(Protocol):
    """Public application boundary for canonical record search."""

    async def search(
        self,
        query: str,
        *,
        limit: int,
        filters: dict[str, object],
    ) -> RecordSearchOutcome: ...

    async def async_search(
        self,
        query: str,
        *,
        limit: int,
        filters: dict[str, object],
    ) -> RecordSearchOutcome: ...


class PipelineSearchBoundary:
    """Adapt the canonical pipeline to the application-owned boundary."""

    def __init__(self, pipeline) -> None:
        self._pipeline = pipeline

    async def search(
        self,
        query: str,
        *,
        limit: int,
        filters: dict[str, object],
    ) -> RecordSearchOutcome:
        return await self._pipeline.async_search(
            query,
            limit=limit,
            filters=dict(filters),
        )

    async def async_search(
        self,
        query: str,
        *,
        limit: int,
        filters: dict[str, object],
    ) -> RecordSearchOutcome:
        """Retain the pipeline method name for in-repo compatibility tests."""
        return await self.search(query, limit=limit, filters=filters)


def to_record_search_config(config: SearchConfig) -> RecordSearchConfig:
    """Map supported application settings without calibrating raw RRF scores.

    Deduplication, document limits, and score thresholds remain query policy;
    recency, project uplift, and reranking have no canonical pipeline equivalent.
    """
    return RecordSearchConfig(
        weighted_rrf_enabled=True,
        base_semantic_weight=config.semantic_weight,
        base_keyword_weight=config.keyword_weight,
        base_graph_weight=1.0,
        rerank_budget=config.rerank_budget,
        minimum_score=(
            config.abstention_threshold
            if config.abstention_threshold is not None
            else 0.0
        ),
        adaptive_enabled=False,
    )


class _SentenceTransformerReranker:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._model: Any = None

    def rerank(self, query: str, documents: list[str]) -> list[float]:
        if self._model is None:
            module = import_module("sentence_transformers")
            self._model = module.CrossEncoder(self.model_name)
        scores = self._model.predict([(query, document) for document in documents])
        return [float(score) for score in scores]


def build_reranker(config: SearchConfig):
    if config.reranker_model is None:
        return None
    return _SentenceTransformerReranker(config.reranker_model)


@dataclass(frozen=True)
class SearchQuery:
    query: str
    top_n: int
    top_k: int | None = None
    project_filter: tuple[str, ...] = ()
    source_filter: tuple[str, ...] = ()
    project_context: str | None = None
    excluded_files: frozenset[str] = frozenset()
    min_score: float | None = None
    similarity_threshold: float | None = None
    max_chunks_per_doc: int = 1
    retrieval_mode: str | None = None


@dataclass(frozen=True)
class SearchExecution:
    results: list[ChunkResult]
    compression_stats: CompressionStats
    strategy_stats: SearchStrategyStats
    query_execution_stats: dict[str, object] = field(default_factory=dict)


def _record_project_id(record: Record) -> str | None:
    workspace_id = record.workspace_id
    if workspace_id is not None:
        return workspace_id
    project_id = record.metadata.get("project_id")
    return project_id if isinstance(project_id, str) else None


class ApplicationSearchUseCase:
    """Apply application query policy, execute search, and map results."""

    def __init__(
        self,
        search_kernel: SearchKernelBoundary,
        *,
        documents_roots: Sequence[Path],
        default_min_score: float | None = None,
    ) -> None:
        self._search_kernel = search_kernel
        self._pipeline = search_kernel
        self._documents_roots = tuple(documents_roots)
        self._default_min_score = default_min_score

    async def execute(
        self,
        request: SearchQuery,
        *,
        search: Callable[..., Awaitable[RecordSearchOutcome]] | None = None,
    ) -> SearchExecution:
        filters: dict[str, object] = {}
        effective_min_score = (
            request.min_score
            if request.min_score is not None
            else self._default_min_score
        )
        if request.source_filter:
            filters["source_kinds"] = list(request.source_filter)
        if request.project_filter:
            if len(request.project_filter) == 1:
                filters["workspace_id"] = request.project_filter[0]
            else:
                filters["project_ids"] = list(request.project_filter)
        if request.excluded_files:
            filters["excluded_files"] = sorted(request.excluded_files)
        if effective_min_score is not None:
            filters["min_score"] = effective_min_score
        if request.similarity_threshold is not None:
            filters["similarity_threshold"] = request.similarity_threshold
        filters["max_chunks_per_doc"] = request.max_chunks_per_doc
        if request.retrieval_mode is not None:
            filters["retrieval_mode"] = request.retrieval_mode

        limit = max(20, request.top_n * 4, request.top_k or 0)
        if request.project_filter:
            limit = max(limit, request.top_n * 10)
        if request.max_chunks_per_doc > 0:
            limit = max(limit, request.top_n * max(4, request.max_chunks_per_doc * 2))
        if request.source_filter == ("git_commit",):
            limit = max(limit, request.top_n * 4)

        if search is None:
            outcome = await self._pipeline.async_search(
                request.query,
                limit=limit,
                filters=filters,
            )
        else:
            outcome = await search(request.query, limit=limit, filters=filters)
        filtered_results = [
            result
            for result in outcome.results
            if (
                not request.project_filter
                or _record_project_id(result.record) in request.project_filter
            )
            and (
                effective_min_score is None
                or result.score >= effective_min_score
            )
            and not self._is_excluded(result.record.metadata, request.excluded_files)
        ]
        if request.max_chunks_per_doc > 0:
            counts: dict[str, int] = {}
            limited_results = []
            for result in filtered_results:
                doc_id = str(
                    result.record.metadata.get("doc_id", result.record.source_id)
                )
                if counts.get(doc_id, 0) >= request.max_chunks_per_doc:
                    continue
                counts[doc_id] = counts.get(doc_id, 0) + 1
                limited_results.append(result)
            filtered_results = limited_results

        results = [
            self._to_chunk_result(result.record, result.score, result.provenance)
            for result in filtered_results
        ]
        if request.source_filter == ("git_commit",):
            results = aggregate_commit_results(results)
        results = results[: request.top_n]
        strategy_counts = {"semantic": 0, "keyword": 0, "graph": 0, "tag_expansion": 0}
        for result in results:
            for strategy in result.provenance.strategies if result.provenance else ():
                strategy_name = {"vector": "semantic", "expansion": "tag_expansion"}.get(
                    strategy, strategy
                )
                if strategy_name in strategy_counts:
                    strategy_counts[strategy_name] += 1
        count = len(results)
        return SearchExecution(
            results=results,
            compression_stats=CompressionStats(
                len(outcome.results),
                len(filtered_results),
                len(filtered_results),
                len(filtered_results),
                len(filtered_results),
                count,
                0,
            ),
            strategy_stats=SearchStrategyStats(
                vector_count=strategy_counts["semantic"],
                keyword_count=strategy_counts["keyword"],
                graph_count=strategy_counts["graph"],
                tag_expansion_count=strategy_counts["tag_expansion"],
            ),
            query_execution_stats={
                "degraded": outcome.degraded,
                "failures": [failure.message for failure in outcome.failures],
            },
        )

    @staticmethod
    def _to_chunk_result(
        record: Record,
        score: float,
        provenance: SearchResultProvenance,
    ) -> ChunkResult:
        metadata = dict(record.metadata)
        metadata.setdefault("record_id", record.storage_key)
        metadata.setdefault("workspace_id", record.workspace_id)
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
        excluded_files: frozenset[str],
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
        for root in self._documents_roots:
            try:
                relative = resolved.relative_to(root.resolve())
                candidates.add(str(relative))
                candidates.add(str(relative.with_suffix("")))
            except ValueError:
                continue
        return bool(candidates & excluded_files)


__all__ = [
    "ApplicationSearchUseCase",
    "PipelineSearchBoundary",
    "SearchExecution",
    "SearchKernelBoundary",
    "SearchQuery",
    "build_reranker",
    "to_record_search_config",
]
