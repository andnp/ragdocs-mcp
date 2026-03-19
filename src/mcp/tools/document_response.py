"""Structured response helpers for MCP document tools."""

from __future__ import annotations

import json
from dataclasses import dataclass

from src.mcp.tools.document_request import NormalizedQueryDocumentsRequest
from src.models import ChunkResult, CompressionStats, SearchStrategyStats
from src.search.utils import truncate_content


@dataclass(frozen=True)
class QueryDocumentsScopeEnvelope:
    """Canonical and effective scope metadata for a query_documents response."""

    mode: str
    projects: tuple[str, ...]
    preferred_project: str | None
    applied_filter_projects: tuple[str, ...]
    applied_uplift_project: str | None

    def to_dict(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "projects": list(self.projects),
            "preferred_project": self.preferred_project,
            "applied_filter_projects": list(self.applied_filter_projects),
            "applied_uplift_project": self.applied_uplift_project,
        }


@dataclass(frozen=True)
class QueryDocumentsResultEnvelopeItem:
    """Stable result metadata for a single query_documents hit."""

    rank: int
    chunk_id: str
    doc_id: str
    file_path: str
    header_path: str
    score: float
    content: str
    project_id: str | None = None
    parent_chunk_id: str | None = None

    def to_dict(self, *, include_content: bool = False) -> dict[str, object]:
        result: dict[str, object] = {
            "rank": self.rank,
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "file_path": self.file_path,
            "header_path": self.header_path,
            "score": self.score,
            "project_id": self.project_id,
            "parent_chunk_id": self.parent_chunk_id,
        }
        if include_content:
            result["content"] = self.content
        return result

    def render_text(self, *, query_type: str) -> str:
        rendered_content = (
            truncate_content(self.content, 200)
            if query_type == "factual"
            else self.content
        )
        return (
            f"[{self.rank}] {self.file_path or 'unknown'} "
            f"§ {self.header_path or '(no section)'} ({self.score:.2f})\n"
            f"{rendered_content}"
        )


@dataclass(frozen=True)
class QueryDocumentsResponseEnvelope:
    """Internal typed response envelope for query_documents."""

    result_header: str
    query: str
    query_type: str
    scope: QueryDocumentsScopeEnvelope
    results: tuple[QueryDocumentsResultEnvelopeItem, ...]
    strategy_counts: dict[str, int]
    observed_strategies: tuple[str, ...]
    min_score: float
    max_chunks_per_doc: int
    compression_stats: CompressionStats | None = None

    @property
    def schema_version(self) -> str:
        return "query_documents.response.v1"

    @property
    def results_count(self) -> int:
        return len(self.results)

    @property
    def filtering_occurred(self) -> bool:
        if self.compression_stats is None:
            return False
        return self.compression_stats.original_count > self.compression_stats.after_dedup

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema_version": self.schema_version,
            "result_header": self.result_header,
            "query": self.query,
            "query_type": self.query_type,
            "results_count": self.results_count,
            "scope": self.scope.to_dict(),
            "strategy_counts": self.strategy_counts,
            "observed_strategies": list(self.observed_strategies),
            "results": [result.to_dict() for result in self.results],
        }
        if self.compression_stats is not None:
            payload["compression_stats"] = self.compression_stats.to_dict()
        return payload

    def render_text(self, *, show_stats: bool) -> str:
        results_text = "\n\n".join(
            result.render_text(query_type=self.query_type) for result in self.results
        )

        response = f"# {self.result_header}\n\n{results_text}"

        if show_stats and self.compression_stats is not None:
            stats = self.compression_stats
            stats_parts = [
                f"- Original results: {stats.original_count}",
                f"- After score filter (≥{self.min_score}): {stats.after_threshold}",
                f"- After deduplication: {stats.after_dedup}",
            ]
            if self.max_chunks_per_doc == 1:
                stats_parts.append(
                    f"- After document limit (1 per doc): {stats.after_doc_limit}"
                )
            stats_parts.append(f"- Clusters merged: {stats.clusters_merged}")
            response = (
                f"{response}\n\n# Compression Stats\n\n" + "\n".join(stats_parts)
            )

        metadata_json = json.dumps(self.to_dict(), ensure_ascii=False, indent=2)
        return (
            f"{response}\n\n<!-- query_documents_response_v1\n"
            f"{metadata_json}\n"
            f"-->"
        )


def build_query_documents_response_envelope(
    request: NormalizedQueryDocumentsRequest,
    *,
    query_type: str,
    results: list[ChunkResult],
    strategy_stats: SearchStrategyStats,
    compression_stats: CompressionStats,
    effective_project_context: str | None,
) -> QueryDocumentsResponseEnvelope:
    """Construct a typed response envelope for query_documents."""

    result_items = tuple(
        QueryDocumentsResultEnvelopeItem(
            rank=index,
            chunk_id=result.chunk_id,
            doc_id=result.doc_id,
            file_path=result.file_path,
            header_path=result.header_path,
            score=result.score,
            content=result.content,
            project_id=result.project_id,
            parent_chunk_id=result.parent_chunk_id,
        )
        for index, result in enumerate(results, start=1)
    )

    strategy_counts = _build_strategy_counts(strategy_stats)
    observed_strategies = tuple(
        strategy_name
        for strategy_name, count in strategy_counts.items()
        if count > 0
    )

    return QueryDocumentsResponseEnvelope(
        result_header=request.result_header,
        query=request.query,
        query_type=query_type,
        scope=QueryDocumentsScopeEnvelope(
            mode=request.scope_mode,
            projects=request.scope_projects,
            preferred_project=request.preferred_project,
            applied_filter_projects=tuple(request.project_filter),
            applied_uplift_project=effective_project_context,
        ),
        results=result_items,
        strategy_counts=strategy_counts,
        observed_strategies=observed_strategies,
        min_score=request.min_score,
        max_chunks_per_doc=request.max_chunks_per_doc,
        compression_stats=compression_stats,
    )


def _build_strategy_counts(strategy_stats: SearchStrategyStats) -> dict[str, int]:
    return {
        "semantic": strategy_stats.vector_count or 0,
        "keyword": strategy_stats.keyword_count or 0,
        "graph": strategy_stats.graph_count or 0,
        "tag_expansion": strategy_stats.tag_expansion_count or 0,
    }