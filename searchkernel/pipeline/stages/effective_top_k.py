"""EffectiveTopKStage: factual-query stage-top_k contraction.

Lifted from `SearchOrchestrator._resolve_effective_stage_top_k`. Runs
immediately after `RoutingStage` because the contraction depends on
`query_type` (routing's own output) -- resolving it inside the pipeline,
rather than the orchestrator computing it between two executor slots, is
what lets `RetrieveStage`/`SeedBookkeepingStage`/`TagExpansionStage`/
`GraphExpandStage` all read a single already-resolved `top_k` key.
"""

from __future__ import annotations

from dataclasses import replace

from searchkernel.pipeline.stage import SearchContext
from searchkernel.search.classifier import QueryType

_REQUESTED_TOP_K_KEY = "requested_top_k"
_TOP_N_KEY = "top_n"
_PROJECT_FILTER_KEY = "project_filter"
_QUERY_TYPE_KEY = "query_type"
_TOP_K_KEY = "top_k"

_FACTUAL_QUERY_CONTRACTED_TOP_K_FLOOR = 8
_FACTUAL_QUERY_CONTRACTED_TOP_K_MULTIPLIER = 2
_FACTUAL_QUERY_TOP_N_CONTRACTION_LIMIT = 5


class EffectiveTopKStage:
    """Contract `top_k` for narrow, high-confidence factual queries.

    Expects `context.metadata["requested_top_k"]` (`int`), `["top_n"]`
    (`int`), `["project_filter"]` (`list[str] | None`) and `["query_type"]`
    (`QueryType`, from `RoutingStage`). Writes `["top_k"]` (`int`), the
    value downstream retrieval/expansion stages consume.
    """

    name = "effective_top_k"

    def run(self, context: SearchContext) -> SearchContext:
        requested_top_k = context.metadata[_REQUESTED_TOP_K_KEY]
        top_n = context.metadata[_TOP_N_KEY]
        project_filter = context.metadata.get(_PROJECT_FILTER_KEY)
        query_type = context.metadata[_QUERY_TYPE_KEY]

        metadata = dict(context.metadata)
        metadata[_TOP_K_KEY] = self._resolve(
            requested_top_k, top_n, project_filter, query_type
        )
        return replace(context, metadata=metadata)

    @staticmethod
    def _resolve(
        requested_top_k: int,
        top_n: int,
        project_filter: list[str] | None,
        query_type: QueryType,
    ) -> int:
        if requested_top_k <= 0:
            return requested_top_k

        if project_filter:
            return requested_top_k

        if query_type is not QueryType.FACTUAL:
            return requested_top_k

        if top_n > _FACTUAL_QUERY_TOP_N_CONTRACTION_LIMIT:
            return requested_top_k

        contracted_top_k = max(
            _FACTUAL_QUERY_CONTRACTED_TOP_K_FLOOR,
            top_n * _FACTUAL_QUERY_CONTRACTED_TOP_K_MULTIPLIER,
        )
        return min(requested_top_k, contracted_top_k)
