"""ProvenanceStage: builds per-chunk `SearchResultProvenance` from strategy results.

Lifted from `SearchOrchestrator._build_result_provenance`. Pure function of
`context.strategy_results` -- for each strategy's ranked
`(chunk_id, raw_score)` list, records the chunk's rank + raw score under
that strategy. Downstream stages/orchestrator glue (community-boost,
project-uplift, parent-expansion) mutate the resulting
`SearchResultProvenance` objects in place, exactly as before this
extraction.
"""

from __future__ import annotations

from dataclasses import replace

from searchkernel.models import SearchResultProvenance
from searchkernel.pipeline.stage import SearchContext

_RESULT_PROVENANCE_KEY = "result_provenance"


class ProvenanceStage:
    """Build `context.metadata["result_provenance"]` from `context.strategy_results`.

    Expects `context.strategy_results` (`dict[str, list[tuple[str, float]]]`,
    keyed by strategy name). Writes
    `context.metadata["result_provenance"]` (`dict[str, SearchResultProvenance]`).
    """

    name = "provenance"

    def run(self, context: SearchContext) -> SearchContext:
        result_provenance: dict[str, SearchResultProvenance] = {}

        for strategy, result_list in context.strategy_results.items():
            for rank, (chunk_id, raw_score) in enumerate(result_list, start=1):
                provenance = result_provenance.setdefault(
                    chunk_id,
                    SearchResultProvenance(),
                )
                provenance.add_strategy(strategy, rank, raw_score)

        metadata = dict(context.metadata)
        metadata[_RESULT_PROVENANCE_KEY] = result_provenance
        return replace(context, metadata=metadata)
