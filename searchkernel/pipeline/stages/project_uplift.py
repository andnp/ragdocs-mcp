"""ProjectUpliftStage: boost candidates matching the active project.

Lifted from `SearchOrchestrator._apply_project_uplift`. Parameterized
over a `get_chunk` callable (the orchestrator's
`QueryExecutionContext.get_vector_chunk` or the bare
`VectorIndex.get_chunk_by_id`) so it composes over either, exactly as
the orchestrator method did before extraction.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from typing import Any

from searchkernel.domain import SearchResultProvenance
from searchkernel.pipeline.stage import SearchContext

GetChunk = Callable[[str], "dict[str, Any] | None"]

_ACTIVE_PROJECT_KEY = "active_project"
_RESULT_PROVENANCE_KEY = "result_provenance"


class ProjectUpliftStage:
    """Boost candidates whose chunk metadata matches the active project.

    Expects `context.candidates` (`list[tuple[chunk_id, score]]`) and
    `context.metadata["active_project"]` (`str | None`). Optionally
    mutates `["result_provenance"]` (`dict[str, SearchResultProvenance]`)
    in place, recording the applied uplift. Writes `context.candidates`
    re-sorted descending by boosted score. A falsy `active_project`
    leaves `context` unchanged.
    """

    name = "project_uplift"

    def __init__(self, get_chunk: GetChunk, uplift: float):
        self._get_chunk = get_chunk
        self._uplift = uplift

    def run(self, context: SearchContext) -> SearchContext:
        active_project = context.metadata.get(_ACTIVE_PROJECT_KEY)
        if not active_project:
            return context

        fused = context.candidates
        result_provenance = context.metadata.get(_RESULT_PROVENANCE_KEY)

        boosted: list[tuple[str, float]] = []
        for chunk_id, score in fused:
            chunk_data = self._get_chunk(chunk_id)
            metadata = chunk_data.get("metadata", {}) if chunk_data else {}
            project_id = (
                metadata.get("project_id") if isinstance(metadata, dict) else None
            )
            if project_id == active_project:
                if result_provenance is not None:
                    provenance = result_provenance.setdefault(
                        chunk_id, SearchResultProvenance()
                    )
                    provenance.project_uplift = self._uplift
                boosted.append((chunk_id, score * self._uplift))
            else:
                boosted.append((chunk_id, score))

        ranked = sorted(boosted, key=lambda x: x[1], reverse=True)
        return replace(context, candidates=ranked)
