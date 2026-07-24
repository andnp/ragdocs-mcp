"""SourceFilterStage: keep only candidates matching a source-kind filter.

Lifted from `SearchOrchestrator._apply_source_filter`. Parameterized
over a `get_chunk` callable, same injection pattern as
`ProjectUpliftStage`/`ProjectFilterStage`.
"""

from __future__ import annotations

from dataclasses import replace

from searchkernel.pipeline.stage import SearchContext
from searchkernel.pipeline.stages.project_uplift import GetChunk

_SOURCE_FILTER_KEY = "source_filter"


class SourceFilterStage:
    """Filter candidates to those whose `source_kind` is in `source_filter`.

    Expects `context.candidates` (`list[tuple[chunk_id, score]]`) and
    `context.metadata["source_filter"]` (`list[str] | None`). A falsy
    filter leaves `context` unchanged; otherwise writes `context.candidates`
    filtered to matching chunks (order preserved).
    """

    name = "source_filter"

    def __init__(self, get_chunk: GetChunk):
        self._get_chunk = get_chunk

    def run(self, context: SearchContext) -> SearchContext:
        source_filter = context.metadata.get(_SOURCE_FILTER_KEY)
        if not source_filter:
            return context

        allowed_kinds = set(source_filter)
        filtered: list[tuple[str, float]] = []
        for chunk_id, score in context.candidates:
            chunk_data = self._get_chunk(chunk_id)
            metadata = chunk_data.get("metadata", {}) if chunk_data else {}
            source_kind = (
                metadata.get("source_kind") if isinstance(metadata, dict) else None
            )
            if source_kind in allowed_kinds:
                filtered.append((chunk_id, score))

        return replace(context, candidates=filtered)
