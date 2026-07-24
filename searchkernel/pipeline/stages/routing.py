"""RoutingStage: the query-classification + adaptive-weight query stage.

Lifted from SearchOrchestrator.query's direct use of
classify_query/get_adaptive_weights behind the SearchStage contract.
Delegates straight to those functions -- same inputs, same outputs --
so this is a pure extraction with no behavior change.
"""

from __future__ import annotations

from dataclasses import replace

from searchkernel.pipeline.stage import SearchContext
from searchkernel.search.classifier import classify_query, get_adaptive_weights

_BASE_SEMANTIC_WEIGHT_KEY = "base_semantic_weight"
_BASE_KEYWORD_WEIGHT_KEY = "base_keyword_weight"
_BASE_GRAPH_WEIGHT_KEY = "base_graph_weight"
_QUERY_TYPE_KEY = "query_type"
_STRATEGY_WEIGHTS_KEY = "strategy_weights"


class RoutingStage:
    """Classify the query and derive adaptive per-strategy weights.

    Expects `context.metadata` to carry `base_semantic_weight`,
    `base_keyword_weight` and `base_graph_weight` (all `float`). Writes
    `context.metadata["query_type"]` (`QueryType`) and
    `["strategy_weights"]` (`dict[str, float]`, keys `semantic`/
    `keyword`/`graph`).
    """

    name = "routing"

    def run(self, context: SearchContext) -> SearchContext:
        base_semantic = context.metadata[_BASE_SEMANTIC_WEIGHT_KEY]
        base_keyword = context.metadata[_BASE_KEYWORD_WEIGHT_KEY]
        base_graph = context.metadata[_BASE_GRAPH_WEIGHT_KEY]

        query_type = classify_query(context.query)
        semantic_w, keyword_w, graph_w = get_adaptive_weights(
            query_type, base_semantic, base_keyword, base_graph
        )

        metadata = dict(context.metadata)
        metadata[_QUERY_TYPE_KEY] = query_type
        metadata[_STRATEGY_WEIGHTS_KEY] = {
            "semantic": semantic_w,
            "keyword": keyword_w,
            "graph": graph_w,
        }
        return replace(context, metadata=metadata)
