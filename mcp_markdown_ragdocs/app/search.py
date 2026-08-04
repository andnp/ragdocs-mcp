"""Application-owned search use case over the canonical record pipeline."""

from __future__ import annotations

import re
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field, fields
from importlib import import_module
from pathlib import Path
from typing import Any, Protocol, cast

from searchkernel.api import (
    CompressionStats,
    Record,
    RecordHit,
    RecordIdentity,
    RecordSearchConfig,
    RecordSearchOutcome,
    RecordSearchPolicy,
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
            _normalize_graph_query(query),
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


def _normalize_graph_query(query: str) -> str:
    return re.sub(
        r"^what\s+links to\s+",
        "Which pages link to ",
        query,
        count=1,
        flags=re.IGNORECASE,
    )


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


_DEFAULT_ABSTENTION_SCORE = 0.01
_DEFAULT_VECTOR_ABSTENTION_SCORE = 0.03
_DEFAULT_HYBRID_KEYWORD_SIGNAL = 0.01
_DEFAULT_STRONG_KEYWORD_SIGNAL = 0.5
_DEFAULT_HYBRID_MAX_KEYWORD_RANK = 5
_DEFAULT_MIN_MEANINGFUL_TOKEN_OVERLAP = 2
_QUERY_STOP_WORDS = frozenset(
    {"a", "an", "and", "are", "be", "for", "how", "in", "is", "must", "of", "the", "to"}
)
_GRAPH_TARGET_PREFIXES = (
    re.compile(
        r"^(?:which|what)\s+(?:pages|documents|notes)\s+"
        r"(?:link to|are neighbors of)\s+",
        re.IGNORECASE,
    ),
    re.compile(
        r"^(?:which|what)\s+(?:pages|documents|notes)\s+are linked from\s+",
        re.IGNORECASE,
    ),
    re.compile(
        r"^show me\s+(?:pages|documents|notes)\s+that\s+(?:link to|embed)\s+",
        re.IGNORECASE,
    ),
    re.compile(r"^what\s+links to\s+", re.IGNORECASE),
)
_GRAPH_TARGET_SUFFIXES = (
    re.compile(
        r"^what\s+(?:pages|documents|notes)\s+does\s+(.+?)\s+link to$",
        re.IGNORECASE,
    ),
)
_GRAPH_INBOUND_PREFIXES = (
    re.compile(
        r"^(?:which|what)\s+(?:pages|documents|notes)\s+link to\s+",
        re.IGNORECASE,
    ),
    re.compile(r"^what\s+links to\s+", re.IGNORECASE),
)
_GRAPH_NEIGHBOR_PREFIX = re.compile(
    r"^(?:which|what)\s+(?:pages|documents|notes)\s+are neighbors of\s+",
    re.IGNORECASE,
)


def _graph_target_query(query: str) -> str:
    normalized = " ".join(query.strip().split()).strip(" .?!")
    normalized = re.sub(
        r"\s+and\s+what\s+do\s+their\s+neighbors?\s+explain$",
        "",
        normalized,
        count=1,
        flags=re.IGNORECASE,
    ).strip(" .?!")
    for suffix in _GRAPH_TARGET_SUFFIXES:
        match = suffix.match(normalized)
        if match is not None:
            return match.group(1).strip(" .?!")
    for prefix in _GRAPH_TARGET_PREFIXES:
        target = prefix.sub("", normalized, count=1).strip(" .?!")
        if target != normalized:
            return target
    return normalized


def _graph_query_is_inbound(query: str) -> bool:
    normalized = " ".join(query.strip().split()).strip(" .?!")
    return any(pattern.match(normalized) is not None for pattern in _GRAPH_INBOUND_PREFIXES)


def _graph_query_direction(query: str) -> str:
    normalized = " ".join(query.strip().split()).strip(" .?!")
    if _graph_query_is_inbound(normalized):
        return "incoming"
    if _GRAPH_NEIGHBOR_PREFIX.match(normalized) is not None:
        return "both"
    return "outgoing"


def build_record_search_policy(
    keyword_store: Any,
    record_hydrator: Any = None,
    graph_direction_setter: Any = None,
) -> RecordSearchPolicy | None:
    """Wire target resolution when the installed kernel exposes the policy hook."""
    if not any(field.name == "graph_target_resolver" for field in fields(RecordSearchPolicy)):
        return None

    async def resolve_graph_target(query: str, context: Any):
        if callable(graph_direction_setter):
            graph_direction_setter(_graph_query_direction(query))
        target_query = _graph_target_query(query)
        filters = dict(getattr(context, "filters", {}))
        limit = max(20, int(getattr(context, "limit", 20)))
        store: Any = keyword_store() if callable(keyword_store) else keyword_store
        hits = store.search(target_query, limit, filters)
        if not callable(record_hydrator):
            normalized_hits = hits
        else:
            normalized: dict[str, RecordHit] = {}
            for hit in hits:
                record = cast(Record | None, record_hydrator(hit.identity))
                parent_chunk_id = (
                    record.metadata.get("parent_chunk_id")
                    if record is not None
                    else None
                )
                identity = (
                    RecordIdentity(
                        hit.workspace_id,
                        hit.source_kind,
                        parent_chunk_id,
                    )
                    if isinstance(parent_chunk_id, str) and parent_chunk_id
                    else hit.identity
                )
                candidate = RecordHit(identity, hit.score)
                previous = normalized.get(candidate.storage_key)
                if previous is None or candidate.score > previous.score:
                    normalized[candidate.storage_key] = candidate
            normalized_hits = sorted(
                normalized.values(),
                key=lambda hit: (-hit.score, hit.storage_key),
            )
        return normalized_hits

    return cast(Any, RecordSearchPolicy)(
        graph_target_resolver=resolve_graph_target
    )


def _record_project_id(record: Record) -> str | None:
    workspace_id = record.workspace_id
    if workspace_id is not None:
        return workspace_id
    project_id = record.metadata.get("project_id")
    return project_id if isinstance(project_id, str) else None


def _document_source_rank(request: SearchQuery, result: Any) -> int:
    """Prefer addressable documents over pathless history in document search."""
    if request.source_filter == ("git_commit",):
        return 0
    record = result.record
    source_kind = record.source_kind or record.metadata.get("source_kind")
    if source_kind == "git_commit" and not (
        record.metadata.get("file_path") or record.uri
    ):
        return 0
    return 1


def _is_pathless_git_record(result: Any) -> bool:
    record = result.record
    source_kind = record.source_kind or record.metadata.get("source_kind")
    return source_kind == "git_commit" and not (
        record.metadata.get("file_path") or record.uri
    )


def _default_match_for_query(query: str, result: Any) -> bool:
    """Keep low-score records with an application-visible lexical signal."""
    record = result.record
    metadata = record.metadata
    tokens = set(re.findall(r"[a-z0-9_./-]+", query.lower()))
    if not tokens:
        return False
    meaningful_tokens = tokens - _QUERY_STOP_WORDS
    searchable_metadata = " ".join(
        str(value)
        for value in (
            record.title,
            metadata.get("header_path"),
            metadata.get("file_path"),
        )
        if value
    ).lower()
    if query.lower() in searchable_metadata or tokens <= set(
        re.findall(r"[a-z0-9_./-]+", searchable_metadata)
    ):
        return True
    provenance = result.provenance
    strategies = getattr(provenance, "strategies", ()) if provenance else ()
    body_tokens = set(re.findall(r"[a-z0-9_./-]+", record.body.lower()))
    overlap = meaningful_tokens & body_tokens
    if "keyword" not in strategies:
        return False
    if set(strategies) == {"keyword"}:
        return bool(overlap)
    return len(overlap) >= _DEFAULT_MIN_MEANINGFUL_TOKEN_OVERLAP


def _default_result_is_credible(query: str, result: Any) -> bool:
    if _default_match_for_query(query, result):
        return True
    provenance = result.provenance
    strategies = set(getattr(provenance, "strategies", ()) if provenance else ())
    if "graph" in strategies:
        return result.score >= _DEFAULT_ABSTENTION_SCORE
    if {"keyword", "vector"} <= strategies:
        details = getattr(provenance, "strategy_details", {})
        keyword = details.get("keyword") if hasattr(details, "get") else None
        keyword_rank = getattr(keyword, "rank", None)
        return (
            isinstance(keyword_rank, int)
            and keyword_rank <= _DEFAULT_HYBRID_MAX_KEYWORD_RANK
            and getattr(keyword, "raw_score", 0.0)
            >= _DEFAULT_STRONG_KEYWORD_SIGNAL
        )
    if strategies == {"keyword"}:
        return False
    if "vector" in strategies and not {"keyword", "graph"} & set(strategies):
        return result.score >= _DEFAULT_VECTOR_ABSTENTION_SCORE
    return result.score >= _DEFAULT_ABSTENTION_SCORE


def _document_key(record: Record) -> str:
    metadata = record.metadata
    document_id = metadata.get("doc_id")
    if isinstance(document_id, str) and document_id:
        return document_id
    file_path = metadata.get("file_path")
    if isinstance(file_path, str) and file_path:
        return f"file:{file_path}"
    return record.source_id


def _metadata_query_rank(query: str, result: Any) -> int:
    record = result.record
    metadata = record.metadata
    query_text = query.lower().strip()
    if not query_text:
        return 0
    fields = (
        str(record.title or ""),
        str(metadata.get("header_path") or ""),
        str(metadata.get("file_path") or ""),
    )
    if any(query_text == field.lower() or query_text in field.lower() for field in fields):
        return 2
    tokens = set(re.findall(r"[a-z0-9_./-]+", query_text))
    field_tokens = set(re.findall(r"[a-z0-9_./-]+", " ".join(fields).lower()))
    return 1 if tokens and tokens <= field_tokens else 0


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
            and (
                request.source_filter == ("git_commit",)
                or not _is_pathless_git_record(result)
            )
        ]
        if (
            request.min_score is None
            and self._default_min_score is None
            and request.source_filter != ("git_commit",)
        ):
            filtered_results = [
                result
                for result in filtered_results
                if _default_result_is_credible(request.query, result)
            ]
        if request.source_filter != ("git_commit",):
            filtered_results.sort(
                key=lambda result: (
                    result.score,
                    _document_source_rank(request, result),
                    _metadata_query_rank(request.query, result),
                ),
                reverse=True,
            )
        if request.max_chunks_per_doc > 0:
            counts: dict[str, int] = {}
            limited_results = []
            for result in filtered_results:
                doc_id = _document_key(result.record)
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
        metadata.setdefault("title", record.title)
        metadata.setdefault("workspace_id", record.workspace_id)
        metadata.setdefault("source_kind", record.source_kind)
        metadata.setdefault("source_id", record.source_id)
        project_id = _record_project_id(record)
        metadata.setdefault("project_id", project_id)
        file_path = str(metadata.get("file_path") or "")
        if not file_path and record.uri:
            file_path = record.uri.removeprefix("file://")
        header_path = str(metadata.get("header_path") or record.title or "")
        return ChunkResult(
            chunk_id=str(metadata.get("chunk_id", record.source_id)),
            doc_id=str(metadata.get("doc_id", record.source_id)),
            score=score,
            header_path=header_path,
            file_path=file_path,
            project_id=project_id,
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
