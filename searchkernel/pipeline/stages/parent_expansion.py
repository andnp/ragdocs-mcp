"""ParentExpansionStage: expand child chunks to their parent chunk.

Lifted from `SearchOrchestrator._expand_to_parents`. Parameterized over
two `get_chunk`-shaped callables (regular chunk lookup vs. the
parent-specific lookup, which the orchestrator's `QueryExecutionContext`
tracks under separate stats counters) so it composes over either
`QueryExecutionContext` or the bare `VectorIndex`, exactly as the
orchestrator method did before extraction.

Stays pure with respect to `context`: chunks/parents that fail to
resolve are reported via `context.metadata["missing_chunk_ids"]`/
`["missing_parent_chunk_ids"]` rather than queuing a reindex directly --
same pattern as `HydrateStage`. The caller (currently the orchestrator)
still owns deciding what to do about missing chunks.
"""

from __future__ import annotations

from dataclasses import replace

from searchkernel.pipeline.stage import SearchContext
from searchkernel.pipeline.stages.project_uplift import GetChunk

_RESULT_PROVENANCE_KEY = "result_provenance"
_MISSING_CHUNK_IDS_KEY = "missing_chunk_ids"
_MISSING_PARENT_CHUNK_IDS_KEY = "missing_parent_chunk_ids"


class ParentExpansionStage:
    """Expand child chunks in `context.candidates` to their parent chunk.

    Expects `context.candidates` (`list[tuple[chunk_id, score]]`) and
    optionally `context.metadata["result_provenance"]`
    (`dict[str, SearchResultProvenance]`, mutated in place for expanded
    parents). Writes `context.candidates` (child ids replaced by parent
    ids where a parent exists, deduped), `["missing_chunk_ids"]` and
    `["missing_parent_chunk_ids"]` (`list[str]`, lookup failures).
    """

    name = "parent_expansion"

    def __init__(self, get_chunk: GetChunk, get_parent_chunk: GetChunk):
        self._get_chunk = get_chunk
        self._get_parent_chunk = get_parent_chunk

    def run(self, context: SearchContext) -> SearchContext:
        result_provenance = context.metadata.get(_RESULT_PROVENANCE_KEY)

        seen_parents: set[str] = set()
        expanded: list[tuple[str, float]] = []
        missing_chunk_ids: list[str] = []
        missing_parent_chunk_ids: list[str] = []

        for chunk_id, score in context.candidates:
            chunk_data = self._get_chunk(chunk_id)
            if not chunk_data:
                missing_chunk_ids.append(chunk_id)
                expanded.append((chunk_id, score))
                continue

            metadata = chunk_data.get("metadata", {})
            parent_chunk_id = (
                metadata.get("parent_chunk_id") if isinstance(metadata, dict) else None
            )
            if not parent_chunk_id:
                expanded.append((chunk_id, score))
                continue

            parent_chunk_id_str = str(parent_chunk_id)
            parent_chunk = self._get_parent_chunk(parent_chunk_id_str)
            if parent_chunk is None:
                missing_parent_chunk_ids.append(parent_chunk_id_str)
                expanded.append((chunk_id, score))
                continue

            if parent_chunk_id_str not in seen_parents:
                seen_parents.add(parent_chunk_id_str)
                if result_provenance is not None:
                    source_provenance = result_provenance.get(chunk_id)
                    if source_provenance is not None:
                        parent_provenance = source_provenance.clone()
                        parent_provenance.parent_expanded_from = chunk_id
                        result_provenance[parent_chunk_id_str] = parent_provenance
                expanded.append((parent_chunk_id_str, score))

        metadata_out = dict(context.metadata)
        metadata_out[_MISSING_CHUNK_IDS_KEY] = missing_chunk_ids
        metadata_out[_MISSING_PARENT_CHUNK_IDS_KEY] = missing_parent_chunk_ids
        return replace(context, candidates=expanded, metadata=metadata_out)
