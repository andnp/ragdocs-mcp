"""HydrateStage: materializes final (chunk_id, score) candidates into ChunkResults.

Lifted from `SearchOrchestrator._materialize_chunk_results`. Parameterized
over a `hydrate_chunk_result` callable (the orchestrator's
`QueryExecutionContext.hydrate_chunk_result` or the bare
`ChunkHydrator.hydrate_chunk_result`) so it composes over either, exactly
as the orchestrator method did before extraction.

Chunks that fail to hydrate get a placeholder `ChunkResult` (empty
content/paths) so callers still see one result per candidate -- same as
before. This stage stays pure with respect to `context`: it reports
failed-hydration chunk ids via `context.metadata["missing_chunk_ids"]`
rather than queuing a reindex itself; the caller (currently the
orchestrator) still owns deciding what to do about missing chunks.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from typing import TYPE_CHECKING

from searchkernel.pipeline.stage import SearchContext
from searchkernel.search.path_utils import extract_doc_id_from_chunk_id

if TYPE_CHECKING:
    from searchkernel.domain import ChunkResult

HydrateChunkResult = Callable[[str, float], "ChunkResult | None"]

_RESULT_PROVENANCE_KEY = "result_provenance"
_CHUNK_RESULTS_KEY = "chunk_results"
_MISSING_CHUNK_IDS_KEY = "missing_chunk_ids"


class HydrateStage:
    """Turn `context.candidates` into `ChunkResult`s.

    Expects `context.candidates` (`list[tuple[chunk_id, score]]`) and
    optionally `context.metadata["result_provenance"]`
    (`dict[str, SearchResultProvenance]`). Writes
    `context.metadata["chunk_results"]` (`list[ChunkResult]`, one per
    candidate, same order) and `["missing_chunk_ids"]` (`list[str]`, the
    chunk ids that failed to hydrate).
    """

    name = "hydrate"

    def __init__(self, hydrate_chunk_result: HydrateChunkResult):
        self._hydrate_chunk_result = hydrate_chunk_result

    def run(self, context: SearchContext) -> SearchContext:
        from searchkernel.domain import ChunkResult

        result_provenance = context.metadata.get(_RESULT_PROVENANCE_KEY)
        chunk_results: list[ChunkResult] = []
        missing_chunk_ids: list[str] = []

        for chunk_id, score in context.candidates:
            chunk_result = self._hydrate_chunk_result(chunk_id, score)
            if chunk_result is not None:
                if result_provenance is not None:
                    chunk_result.provenance = result_provenance.get(chunk_id)
                chunk_results.append(chunk_result)
                continue

            missing_chunk_ids.append(chunk_id)
            chunk_results.append(
                ChunkResult(
                    chunk_id=chunk_id,
                    record_id=extract_doc_id_from_chunk_id(chunk_id),
                    score=score,
                    content="",
                    metadata={"header_path": "", "file_path": ""},
                    provenance=(
                        result_provenance.get(chunk_id)
                        if result_provenance is not None
                        else None
                    ),
                )
            )

        metadata = dict(context.metadata)
        metadata[_CHUNK_RESULTS_KEY] = chunk_results
        metadata[_MISSING_CHUNK_IDS_KEY] = missing_chunk_ids
        return replace(context, metadata=metadata)
