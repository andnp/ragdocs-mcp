"""Application-owned search use case over the canonical record pipeline."""

from __future__ import annotations

import re
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field, fields
from importlib import import_module
from pathlib import Path
from typing import Any, cast

from searchkernel.api import (
    CompressionStats,
    Record,
    RecordHit,
    RecordIdentity,
    RecordSearchConfig,
    RecordSearchOutcome,
    RecordSearchPolicy,
    SearchStrategyStats,
)
try:
    from searchkernel.api import parse_relationship_intent
except ImportError:
    parse_relationship_intent = None

from mcp_markdown_ragdocs.config import SearchConfig
from mcp_markdown_ragdocs.app.contracts import (
    Reranker,
    SearchDiagnosticsPort,
    SearchExecutionPort,
)
from mcp_markdown_ragdocs.git.results import aggregate_commit_results
from mcp_markdown_ragdocs.models import ChunkResult
from mcp_markdown_ragdocs.app.searchkernel_adapter import (
    _record_project_id,
    build_search_diagnostics,
    map_kernel_result,
)


class PipelineSearchBoundary:
    """Adapt the canonical pipeline to the application-owned boundary."""

    def __init__(self, pipeline: SearchExecutionPort) -> None:
        self._pipeline = pipeline

    async def search(
        self,
        query: str,
        *,
        limit: int,
        filters: Mapping[str, object],
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
        filters: Mapping[str, object],
    ) -> RecordSearchOutcome:
        """Retain the pipeline method name for in-repo compatibility tests."""
        return await self.search(query, limit=limit, filters=filters)


def _normalize_graph_query(query: str) -> str:
    query = re.sub(
        r"^what\s+links to\s+",
        "Which pages link to ",
        query,
        count=1,
        flags=re.IGNORECASE,
    )
    return re.sub(r"\b([a-zA-Z0-9]+)-([a-zA-Z0-9]+)\b", r"\1-\2 \1\2", query)







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


def build_reranker(config: SearchConfig) -> Reranker | None:
    if config.reranker_model is None:
        return None
    return _SentenceTransformerReranker(config.reranker_model)


@dataclass(frozen=True)
class SearchRequest:
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


SearchQuery = SearchRequest


@dataclass(frozen=True)
class SearchExecution:
    results: list[ChunkResult]
    compression_stats: CompressionStats
    strategy_stats: SearchStrategyStats
    query_execution_stats: dict[str, object] = field(default_factory=dict)


_record_outcome_diagnostics = build_search_diagnostics


_DEFAULT_ABSTENTION_SCORE = 0.01
_DEFAULT_VECTOR_ABSTENTION_SCORE = 0.03
_DEFAULT_HYBRID_KEYWORD_SIGNAL = 0.01
_DEFAULT_STRONG_KEYWORD_SIGNAL = 0.5
_DEFAULT_HYBRID_MAX_KEYWORD_RANK = 5
_DEFAULT_MIN_MEANINGFUL_TOKEN_OVERLAP = 2
_SEARCH_TOKEN_PATTERN = re.compile(r"[a-z0-9_/\\-]+(?:\.[a-z0-9_/\\-]+)*")
_QUERY_STOP_WORDS = frozenset(
    {
        "a", "an", "and", "are", "be", "do", "does", "for", "how", "in", "is",
        "link", "links", "must", "neighbor", "neighbors", "notes", "of", "pages",
        "the", "to", "what", "which", "depend", "depends", "documents",
    }
)
_GRAPH_TARGET_PREFIXES = (
    re.compile(
        r"^(?:which|what)\s+(?:pages|documents|notes)\s+"
        r"(?:link to(?: or depend on)?|depend on|are neighbors of)\s+",
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
        r"^what\s+(?:pages|documents|notes)\s+does\s+(.+?)\s+"
        r"(?:link to|depend on|point to|embed|transclude)$",
        re.IGNORECASE,
    ),
)


def _search_tokens(value: str) -> set[str]:
    return set(_SEARCH_TOKEN_PATTERN.findall(value.lower()))
_GRAPH_INBOUND_PREFIXES = (
    re.compile(
        r"^(?:which|what)\s+(?:pages|documents|notes)\s+link to\s+",
        re.IGNORECASE,
    ),
    re.compile(
        r"^(?:which|what)\s+(?:pages|documents|notes)\s+"
        r"(?:are linked from|depend on|link to(?: or depend on)?)\s+",
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
    if parse_relationship_intent is not None:
        intent = parse_relationship_intent(normalized)
        if intent is not None:
            return intent.target
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
            return re.sub(r"^(?:the|a|an)\s+", "", target, count=1, flags=re.IGNORECASE)
    return normalized


def _graph_query_is_inbound(query: str) -> bool:
    normalized = " ".join(query.strip().split()).strip(" .?!")
    return any(pattern.match(normalized) is not None for pattern in _GRAPH_INBOUND_PREFIXES)


def _graph_query_direction(query: str) -> str:
    normalized = " ".join(query.strip().split()).strip(" .?!")
    if parse_relationship_intent is not None:
        intent = parse_relationship_intent(normalized)
        if intent is not None:
            return intent.direction
    if _graph_query_is_inbound(normalized):
        return "incoming"
    if _GRAPH_NEIGHBOR_PREFIX.match(normalized) is not None:
        return "both"
    return "outgoing"


def build_record_search_policy(
    keyword_store: Any,
    record_hydrator: Any = None,
    graph_direction_setter: Any = None,
    project_uplift_multiplier: float = 1.2,
) -> RecordSearchPolicy | None:
    """Wire application-owned query policy into the canonical pipeline."""
    policy_fields = {field.name for field in fields(RecordSearchPolicy)}
    if not {"graph_target_resolver", "query_score_adjuster"} & policy_fields:
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

    def adjust_project_score(candidate: Any, context: Any) -> float:
        preferred_project = getattr(context, "filters", {}).get(
            "ranking_workspace_id"
        )
        if not isinstance(preferred_project, str) or (
            candidate.workspace_id != preferred_project
        ):
            return candidate.score
        multiplier = getattr(context, "filters", {}).get(
            "project_uplift_multiplier", project_uplift_multiplier
        )
        if not isinstance(multiplier, (int, float)):
            multiplier = project_uplift_multiplier
        return min(candidate.score * max(1.0, float(multiplier)), 1.0)

    policy_kwargs: dict[str, Any] = {}
    if "graph_target_resolver" in policy_fields:
        policy_kwargs["graph_target_resolver"] = resolve_graph_target
    if "query_score_adjuster" in policy_fields:
        policy_kwargs["query_score_adjuster"] = adjust_project_score
    return cast(Any, RecordSearchPolicy)(**policy_kwargs)


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


def _artifact_query_matches_git_record(query: str, record: Record) -> bool:
    tokens = [
        token
        for token in _search_tokens(query)
        if any(separator in token for separator in ("_", ".", "/", "\\", "-"))
    ]
    if not tokens:
        return False
    changed_files = record.metadata.get("files_changed")
    changed_file_values = changed_files if isinstance(changed_files, list) else ()
    searchable = " ".join(
        [
            record.title,
            record.body,
            *(str(path) for path in changed_file_values),
        ]
    ).lower()
    return any(token in searchable for token in tokens)


def _default_match_for_query(query: str, result: Any) -> bool:
    """Keep low-score records with an application-visible lexical signal."""
    record = result.record
    metadata = record.metadata
    tokens = _search_tokens(query)
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
        _search_tokens(searchable_metadata)
    ):
        return True
    provenance = result.provenance
    strategies = getattr(provenance, "strategies", ()) if provenance else ()
    body_tokens = _search_tokens(record.body)
    overlap = meaningful_tokens & body_tokens
    if "keyword" not in strategies:
        def _fuzzy_token_match(q_token: str, target_tokens: set[str]) -> bool:
            if len(q_token) < 4:
                return False
            prefix = q_token[:4]
            return any(t.startswith(prefix) for t in target_tokens if len(t) >= 4)

        fuzzy_matches = {
            q for q in meaningful_tokens if _fuzzy_token_match(q, body_tokens)
        }
        return len(overlap | fuzzy_matches) >= _DEFAULT_MIN_MEANINGFUL_TOKEN_OVERLAP
    if set(strategies) == {"keyword"}:
        return bool(overlap)
    required_overlap = min(
        _DEFAULT_MIN_MEANINGFUL_TOKEN_OVERLAP,
        len(meaningful_tokens),
    )
    if required_overlap < _DEFAULT_MIN_MEANINGFUL_TOKEN_OVERLAP and _is_relationship_query_subject(
        query, record
    ):
        required_overlap = _DEFAULT_MIN_MEANINGFUL_TOKEN_OVERLAP
    return len(overlap) >= required_overlap


def _is_relationship_query_subject(query: str, record: Record) -> bool:
    """Is `record` the document a relationship query asks about, not an answer.

    The single-token overlap relaxation lets a relationship query's own
    subject (e.g. "target" in "what links to target") slip back into results
    on nothing more than its own title or body mentioning the query text.
    That subject is not a graph neighbor and should not gain credibility
    through the same relaxation meant for genuine single-word content
    matches.
    """
    if not _is_relationship_query(query):
        return False
    target = _graph_target_query(query).strip().lower()
    if not target:
        return False
    metadata = record.metadata
    candidates = {str(record.title or "").strip().lower()}
    file_path = metadata.get("file_path")
    if isinstance(file_path, str) and file_path:
        candidates.add(Path(file_path).stem.lower())
    doc_id = metadata.get("doc_id")
    if isinstance(doc_id, str) and doc_id:
        candidates.add(doc_id.lower())
        candidates.add(Path(doc_id).stem.lower())
    return target in candidates


def _is_relationship_query(query: str) -> bool:
    if parse_relationship_intent is not None:
        return parse_relationship_intent(query) is not None
    return bool(
        re.search(
            r"\b(?:link|links|linked|depend|depends|dependency|neighbor|"
            r"neighbors|embed|embeds|transclude|transcludes|inbound|outbound)\b",
            query,
            re.IGNORECASE,
        )
    )



def _default_result_is_credible(query: str, result: Any) -> bool:
    if _artifact_query_matches_git_record(query, result.record):
        return True
    if _default_match_for_query(query, result):
        return True
    provenance = result.provenance
    strategies = set(getattr(provenance, "strategies", ()) if provenance else ())
    if "graph" in strategies and _is_relationship_query(query):
        return True
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
    tokens = _search_tokens(query_text)
    field_tokens = _search_tokens(" ".join(fields))
    return 1 if tokens and tokens <= field_tokens else 0


class ApplicationSearchUseCase:
    """Apply application query policy, execute search, and map results."""

    def __init__(
        self,
        search_kernel: SearchExecutionPort,
        *,
        documents_roots: Sequence[Path],
        default_min_score: float | None = None,
        project_uplift_multiplier: float = 1.2,
        diagnostics: SearchDiagnosticsPort = build_search_diagnostics,
    ) -> None:
        self._search_kernel = search_kernel
        self._pipeline = search_kernel
        self._documents_roots = tuple(documents_roots)
        self._default_min_score = default_min_score
        self._project_uplift_multiplier = project_uplift_multiplier
        self._diagnostics = diagnostics

    async def execute(
        self,
        request: SearchRequest,
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
        if request.project_context is not None:
            filters["ranking_workspace_id"] = request.project_context
            filters["project_uplift_multiplier"] = self._project_uplift_multiplier
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
                or _artifact_query_matches_git_record(request.query, result.record)
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

        if request.source_filter != ("git_commit",):
            header_counts: dict[str, int] = {}
            diverse_results = []
            for result in filtered_results:
                meta = dict(result.record.metadata)
                hpath = str(meta.get("header_path") or result.record.title or "")
                count = header_counts.get(hpath, 0)
                header_counts[hpath] = count + 1
                decay = 0.1 ** count
                adjusted_rank_score = result.score * decay
                diverse_results.append((adjusted_rank_score, result))
            diverse_results.sort(
                key=lambda item: (
                    item[0],
                    _document_source_rank(request, item[1]),
                    _metadata_query_rank(request.query, item[1]),
                ),
                reverse=True,
            )
            filtered_results = [item[1] for item in diverse_results]



        results = [
            map_kernel_result(result)
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
            query_execution_stats=self._diagnostics(outcome),
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
    "SearchDiagnosticsPort",
    "SearchExecutionPort",
    "SearchQuery",
    "SearchRequest",
    "Reranker",
    "build_search_diagnostics",
    "map_kernel_result",
    "build_reranker",
    "to_record_search_config",
]
