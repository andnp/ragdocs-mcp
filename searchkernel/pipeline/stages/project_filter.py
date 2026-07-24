"""ProjectFilterStage: keep only candidates matching a project filter.

Lifted from `SearchOrchestrator._apply_project_filter`. Parameterized
over a `get_chunk` callable, same injection pattern as
`ProjectUpliftStage`.
"""

from __future__ import annotations

from dataclasses import replace

from searchkernel.pipeline.stage import SearchContext
from searchkernel.pipeline.stages.project_uplift import GetChunk
from searchkernel.search.filters import matches_project_filter, normalize_project_filter

_PROJECT_FILTER_KEY = "project_filter"


class ProjectFilterStage:
    """Filter candidates to those whose project matches `project_filter`.

    Expects `context.candidates` (`list[tuple[chunk_id, score]]`) and
    `context.metadata["project_filter"]`
    (`list[str] | tuple[str, ...] | set[str] | None`). A `None`/empty
    filter leaves `context` unchanged; otherwise writes `context.candidates`
    filtered to matching chunks (order preserved).
    """

    name = "project_filter"

    def __init__(self, get_chunk: GetChunk):
        self._get_chunk = get_chunk

    def run(self, context: SearchContext) -> SearchContext:
        normalized_filter = normalize_project_filter(
            context.metadata.get(_PROJECT_FILTER_KEY)
        )
        if normalized_filter is None:
            return context

        filtered: list[tuple[str, float]] = []
        for chunk_id, score in context.candidates:
            chunk_data = self._get_chunk(chunk_id)
            metadata = chunk_data.get("metadata", {}) if chunk_data else {}
            project_id = (
                metadata.get("project_id") if isinstance(metadata, dict) else None
            )
            if matches_project_filter(
                str(project_id) if project_id is not None else None,
                normalized_filter,
            ):
                filtered.append((chunk_id, score))

        return replace(context, candidates=filtered)
