"""Structured response helpers for MCP document tools."""

from __future__ import annotations

import json
from dataclasses import dataclass

from src.mcp.tools.document_request import NormalizedQueryDocumentsRequest
from src.models import ChunkResult, CompressionStats, SearchStrategyStats


@dataclass(frozen=True)
class QueryDocumentsScopeEnvelope:
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
    rank: int
    chunk_id: str
    doc_id: str
    file_path: str
    header_path: str
    score: float
    content: str
    project_id: str | None = None
    parent_chunk_id: str | None = None

    def to_dict(self) -> dict[str, object]:
        result: dict[str, object] = {
            "rank": self.rank,
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "file_path": self.file_path,
            "header_path": self.header_path,
            "score": self.score,
            "content": self.content,
            "project_id": self.project_id,
            "parent_chunk_id": self.parent_chunk_id,
        }
        return result


@dataclass(frozen=True)
class QueryDocumentsMetaEnvelope:
    query: str
    query_type: str | None
    scope: QueryDocumentsScopeEnvelope
    results_count: int
    uniqueness_mode: str
    min_score: float | None = None
    similarity_threshold: float | None = None
    strategy_counts: dict[str, int] | None = None
    observed_strategies: tuple[str, ...] = ()
    compression: CompressionStats | None = None
    message: str | None = None
    error: str | None = None
    lifecycle: str | None = None
    daemon_scope: str | None = None
    project_context_mode: str | None = None
    configured_root_count: int | None = None
    index_state: dict[str, object] | None = None

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "query": self.query,
            "scope": self.scope.to_dict(),
            "results_count": self.results_count,
            "uniqueness_mode": self.uniqueness_mode,
        }
        if self.query_type is not None:
            payload["query_type"] = self.query_type
        if self.min_score is not None:
            payload["min_score"] = self.min_score
        if self.similarity_threshold is not None:
            payload["similarity_threshold"] = self.similarity_threshold
        if self.strategy_counts is not None:
            payload["strategy_counts"] = self.strategy_counts
        if self.observed_strategies:
            payload["observed_strategies"] = list(self.observed_strategies)
        if self.compression is not None:
            payload["compression"] = self.compression.to_dict()
        if self.message is not None:
            payload["message"] = self.message
        if self.error is not None:
            payload["error"] = self.error
        if self.lifecycle is not None:
            payload["lifecycle"] = self.lifecycle
        if self.daemon_scope is not None:
            payload["daemon_scope"] = self.daemon_scope
        if self.project_context_mode is not None:
            payload["project_context_mode"] = self.project_context_mode
        if self.configured_root_count is not None:
            payload["configured_root_count"] = self.configured_root_count
        if self.index_state is not None:
            payload["index_state"] = self.index_state
        return payload


@dataclass(frozen=True)
class QueryDocumentsResponseEnvelope:
    status: str
    results: tuple[QueryDocumentsResultEnvelopeItem, ...]
    meta: QueryDocumentsMetaEnvelope

    @property
    def schema_version(self) -> str:
        return "query_documents.response.v2"

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "status": self.status,
            "results": [result.to_dict() for result in self.results],
            "meta": self.meta.to_dict(),
        }

    def render_text(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)


def build_query_documents_response_envelope(
    request: NormalizedQueryDocumentsRequest,
    *,
    query_type: str,
    results: list[ChunkResult],
    strategy_stats: SearchStrategyStats,
    compression_stats: CompressionStats,
    effective_project_context: str | None,
) -> QueryDocumentsResponseEnvelope:
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
        status="ok",
        results=result_items,
        meta=QueryDocumentsMetaEnvelope(
            query=request.query,
            query_type=query_type,
            scope=QueryDocumentsScopeEnvelope(
                mode=request.scope_mode,
                projects=request.scope_projects,
                preferred_project=request.preferred_project,
                applied_filter_projects=tuple(request.project_filter),
                applied_uplift_project=effective_project_context,
            ),
            results_count=len(result_items),
            uniqueness_mode=request.uniqueness_mode,
            min_score=request.min_score,
            similarity_threshold=request.similarity_threshold,
            strategy_counts=strategy_counts,
            observed_strategies=observed_strategies,
            compression=compression_stats,
        ),
    )


def build_query_documents_status_envelope(
    request: NormalizedQueryDocumentsRequest | None,
    *,
    status: str,
    payload: dict[str, object],
) -> QueryDocumentsResponseEnvelope:
    scope = QueryDocumentsScopeEnvelope(
        mode=request.scope_mode if request is not None else "global",
        projects=request.scope_projects if request is not None else (),
        preferred_project=request.preferred_project if request is not None else None,
        applied_filter_projects=tuple(request.project_filter) if request is not None else (),
        applied_uplift_project=None,
    )
    return QueryDocumentsResponseEnvelope(
        status=status,
        results=(),
        meta=QueryDocumentsMetaEnvelope(
            query=request.query if request is not None else str(payload.get("query", "")),
            query_type=None,
            scope=scope,
            results_count=0,
            uniqueness_mode=(
                request.uniqueness_mode if request is not None else "allow_multiple"
            ),
            min_score=request.min_score if request is not None else None,
            similarity_threshold=(
                request.similarity_threshold if request is not None else None
            ),
            message=(
                str(payload["message"])
                if isinstance(payload.get("message"), str)
                else None
            ),
            error=str(payload["error"]) if isinstance(payload.get("error"), str) else None,
            lifecycle=(
                str(payload["lifecycle"])
                if isinstance(payload.get("lifecycle"), str)
                else None
            ),
            daemon_scope=(
                str(payload["daemon_scope"])
                if isinstance(payload.get("daemon_scope"), str)
                else None
            ),
            project_context_mode=(
                str(payload["project_context_mode"])
                if isinstance(payload.get("project_context_mode"), str)
                else None
            ),
            configured_root_count=(
                int(payload["configured_root_count"])
                if payload.get("configured_root_count") is not None
                else None
            ),
            index_state=(
                payload["index_state"]
                if isinstance(payload.get("index_state"), dict)
                else None
            ),
        ),
    )


def build_query_documents_validation_error(
    *, query: str, message: str
) -> QueryDocumentsResponseEnvelope:
    return QueryDocumentsResponseEnvelope(
        status="error",
        results=(),
        meta=QueryDocumentsMetaEnvelope(
            query=query,
            query_type=None,
            scope=QueryDocumentsScopeEnvelope(
                mode="global",
                projects=(),
                preferred_project=None,
                applied_filter_projects=(),
                applied_uplift_project=None,
            ),
            results_count=0,
            uniqueness_mode="allow_multiple",
            message=message,
            error="validation_error",
        ),
    )


def _build_strategy_counts(strategy_stats: SearchStrategyStats) -> dict[str, int]:
    return {
        "semantic": strategy_stats.vector_count or 0,
        "keyword": strategy_stats.keyword_count or 0,
        "graph": strategy_stats.graph_count or 0,
        "tag_expansion": strategy_stats.tag_expansion_count or 0,
    }