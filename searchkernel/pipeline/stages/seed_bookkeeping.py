"""SeedBookkeepingStage: seed-score and chunk/doc-id bookkeeping.

Lifted from the inline `_build_graph_seed_scores` call plus the
all_doc_ids/chunk_id_to_doc_id accumulation loop in
`SearchOrchestrator.query`, run once retrieval has produced
`vector_results`/`keyword_results` and before tag expansion mutates them.
Also carries the factual-query fast-path gate (previously
`SearchOrchestrator._should_skip_expensive_factual_enrichments`) since it
is a pure function of the same two result lists plus `query_type`,
computed at the same point in the pipeline.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from searchkernel.pipeline.stage import SearchContext
from searchkernel.search.classifier import QueryType

_VECTOR_RESULTS_KEY = "vector_results"
_KEYWORD_RESULTS_KEY = "keyword_results"
_QUERY_TYPE_KEY = "query_type"
_SEED_SCORES_KEY = "seed_scores"
_CHUNK_ID_TO_DOC_ID_KEY = "chunk_id_to_doc_id"
_ALL_DOC_IDS_KEY = "all_doc_ids"
_SKIP_TAG_EXPANSION_KEY = "skip_tag_expansion"

_FACTUAL_QUERY_CLEAR_CANDIDATE_LIMIT = 6
_FACTUAL_QUERY_CONSENSUS_DEPTH = 2


def build_graph_seed_scores(
    vector_results: list[dict[str, Any]],
    keyword_results: list[dict[str, Any]],
) -> dict[str, float]:
    """Best per-doc score across vector + keyword results, for graph seeding."""
    seed_scores: dict[str, float] = {}

    for result in vector_results + keyword_results:
        doc_id_obj = result.get("doc_id")
        if not isinstance(doc_id_obj, str) or not doc_id_obj:
            continue

        raw_score = result.get("score", 0.0)
        score = float(raw_score) if isinstance(raw_score, int | float) else 0.0
        current_score = seed_scores.get(doc_id_obj, 0.0)
        if score > current_score:
            seed_scores[doc_id_obj] = score

    return seed_scores


def should_skip_expensive_factual_enrichments(
    query_type: QueryType,
    vector_results: list[dict[str, Any]],
    keyword_results: list[dict[str, Any]],
) -> bool:
    """Whether a factual query already has a clear, consensus answer."""
    if query_type is not QueryType.FACTUAL:
        return False

    unique_chunk_ids = {
        str(result["chunk_id"])
        for result in vector_results + keyword_results
        if isinstance(result.get("chunk_id"), str) and result.get("chunk_id")
    }
    if len(unique_chunk_ids) <= 1:
        return True
    if len(unique_chunk_ids) > _FACTUAL_QUERY_CLEAR_CANDIDATE_LIMIT:
        return False

    vector_top = [
        str(result["chunk_id"])
        for result in vector_results[:_FACTUAL_QUERY_CONSENSUS_DEPTH]
        if isinstance(result.get("chunk_id"), str) and result.get("chunk_id")
    ]
    keyword_top = [
        str(result["chunk_id"])
        for result in keyword_results[:_FACTUAL_QUERY_CONSENSUS_DEPTH]
        if isinstance(result.get("chunk_id"), str) and result.get("chunk_id")
    ]

    if not vector_top or not keyword_top:
        return False

    if vector_top[0] == keyword_top[0]:
        return True

    return bool(set(vector_top) & set(keyword_top))


class SeedBookkeepingStage:
    """Build graph seed scores and chunk/doc-id bookkeeping from retrieval results.

    Expects `context.metadata["vector_results"]`/`["keyword_results"]`
    (`list[dict]`) and `["query_type"]` (`QueryType`, from `RoutingStage`).
    Writes `["seed_scores"]` (`dict[str, float]`, pre-tag-expansion best
    score per doc id), `["chunk_id_to_doc_id"]` (`dict[str, str]`),
    `["all_doc_ids"]` (`set[str]`) and `["skip_tag_expansion"]` (`bool`).
    """

    name = "seed_bookkeeping"

    def run(self, context: SearchContext) -> SearchContext:
        vector_results = context.metadata[_VECTOR_RESULTS_KEY]
        keyword_results = context.metadata[_KEYWORD_RESULTS_KEY]
        query_type = context.metadata[_QUERY_TYPE_KEY]

        all_doc_ids: set[str] = set()
        chunk_id_to_doc_id: dict[str, str] = {}
        for result in vector_results + keyword_results:
            chunk_id = result["chunk_id"]
            doc_id = result["doc_id"]
            all_doc_ids.add(doc_id)
            chunk_id_to_doc_id[chunk_id] = doc_id

        metadata = dict(context.metadata)
        metadata[_SEED_SCORES_KEY] = build_graph_seed_scores(
            vector_results, keyword_results
        )
        metadata[_CHUNK_ID_TO_DOC_ID_KEY] = chunk_id_to_doc_id
        metadata[_ALL_DOC_IDS_KEY] = all_doc_ids
        metadata[_SKIP_TAG_EXPANSION_KEY] = should_skip_expensive_factual_enrichments(
            query_type, vector_results, keyword_results
        )
        return replace(context, metadata=metadata)
