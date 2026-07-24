"""StrategyResultsStage: assembles per-strategy scored candidate lists.

Lifted from the inline `strategy_results`/`provenance_results` dict
construction in `SearchOrchestrator.query`. Writes the *narrow*
semantic/keyword/graph dict `FusionStage` fuses (as
`context.strategy_results`) and, separately, the *richer* dict that also
carries a `tag_expansion` strategy when tag expansion actually contributed
results (as `context.metadata["provenance_strategy_results"]`) --
`ProvenanceStage` prefers this metadata key over `context.strategy_results`
when present.

The two dicts must differ: `ScorePipeline.fuse` defaults an unlisted
strategy's weight to `1.0`, so folding `tag_expansion` into the fused
dict (rather than just the provenance-only one) would change ranking.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from searchkernel.pipeline.stage import SearchContext
from searchkernel.search.path_utils import extract_doc_id_from_chunk_id

_VECTOR_RESULTS_KEY = "vector_results"
_KEYWORD_RESULTS_KEY = "keyword_results"
_GRAPH_CHUNK_IDS_KEY = "graph_chunk_ids"
_GRAPH_DOC_SCORES_KEY = "graph_doc_scores"
_APPLIED_TAG_EXPANSION_RESULTS_KEY = "applied_tag_expansion_results"
_PROVENANCE_STRATEGY_RESULTS_KEY = "provenance_strategy_results"


class StrategyResultsStage:
    """Build the per-strategy `(chunk_id, score)` lists fusion/provenance consume.

    Expects `context.metadata["vector_results"]`/`["keyword_results"]`
    (`list[dict]`), `["graph_chunk_ids"]` (`list[str]`),
    `["graph_doc_scores"]` (`dict[str, float]`) and optionally
    `["applied_tag_expansion_results"]` (`list[dict]`). Writes
    `context.strategy_results` (`dict[str, list[tuple[str, float]]]`,
    keys `semantic`/`keyword`/`graph`) and
    `context.metadata["provenance_strategy_results"]` (same, plus a
    `tag_expansion` key when the applied list is non-empty).
    """

    name = "strategy_results"

    def run(self, context: SearchContext) -> SearchContext:
        vector_results: list[dict[str, Any]] = context.metadata[_VECTOR_RESULTS_KEY]
        keyword_results: list[dict[str, Any]] = context.metadata[_KEYWORD_RESULTS_KEY]
        graph_chunk_ids: list[str] = context.metadata[_GRAPH_CHUNK_IDS_KEY]
        graph_doc_scores: dict[str, float] = context.metadata[_GRAPH_DOC_SCORES_KEY]
        applied_tag_expansion_results: list[dict[str, Any]] = context.metadata.get(
            _APPLIED_TAG_EXPANSION_RESULTS_KEY, []
        )

        strategy_results: dict[str, list[tuple[str, float]]] = {
            "semantic": [(r["chunk_id"], r.get("score", 0.0)) for r in vector_results],
            "keyword": [(r["chunk_id"], r.get("score", 0.0)) for r in keyword_results],
            "graph": [
                (
                    chunk_id,
                    graph_doc_scores.get(extract_doc_id_from_chunk_id(chunk_id), 0.0),
                )
                for chunk_id in graph_chunk_ids
            ],
        }

        provenance_strategy_results = dict(strategy_results)
        if applied_tag_expansion_results:
            provenance_strategy_results["tag_expansion"] = [
                (r["chunk_id"], r.get("score", 0.0))
                for r in applied_tag_expansion_results
            ]

        metadata = dict(context.metadata)
        metadata[_PROVENANCE_STRATEGY_RESULTS_KEY] = provenance_strategy_results
        return replace(context, strategy_results=strategy_results, metadata=metadata)
