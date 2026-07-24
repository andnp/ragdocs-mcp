"""CommunityBoostStage: graph-community score boost query stage.

Lifted from `SearchOrchestrator._apply_community_boost`. Parameterized
over a `boost_by_community` callable (the orchestrator's own
`self._graph.boost_by_community`) so it composes over the orchestrator's
graph store without importing it directly, staying monkeypatchable in
existing tests -- same pattern as `GraphExpandStage`'s injected
callables.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

from searchkernel.models import SearchResultProvenance
from searchkernel.pipeline.stage import SearchContext

BoostByCommunity = Callable[[list[str], set[str], float], dict[str, float]]

_SEED_DOC_IDS_KEY = "seed_doc_ids"
_CHUNK_ID_TO_DOC_ID_KEY = "chunk_id_to_doc_id"
_RESULT_PROVENANCE_KEY = "result_provenance"

_BOOST_FACTOR = 1.1


class CommunityBoostStage:
    """Boost candidate scores by graph-community membership.

    Expects `context.candidates` (`list[tuple[chunk_id, score]]`),
    `context.metadata["seed_doc_ids"]` (`set[str]`) and
    `["chunk_id_to_doc_id"]` (`dict[str, str]`). Optionally mutates
    `["result_provenance"]` (`dict[str, SearchResultProvenance]`) in
    place, recording the applied boost. Writes `context.candidates`
    re-sorted descending by boosted score.
    """

    name = "community_boost"

    def __init__(self, boost_by_community: BoostByCommunity):
        self._boost_by_community = boost_by_community

    def run(self, context: SearchContext) -> SearchContext:
        fused = context.candidates
        seed_doc_ids = context.metadata[_SEED_DOC_IDS_KEY]
        chunk_id_to_doc_id = context.metadata[_CHUNK_ID_TO_DOC_ID_KEY]
        result_provenance = context.metadata.get(_RESULT_PROVENANCE_KEY)

        chunk_doc_ids = []
        for chunk_id, _score in fused:
            doc_id = chunk_id_to_doc_id.get(chunk_id)
            if doc_id is None:
                doc_id = (
                    chunk_id.rsplit("_chunk_", 1)[0]
                    if "_chunk_" in chunk_id
                    else chunk_id
                )
            chunk_doc_ids.append(doc_id)

        boosts = self._boost_by_community(chunk_doc_ids, seed_doc_ids, _BOOST_FACTOR)

        boosted = []
        for (chunk_id, score), doc_id in zip(fused, chunk_doc_ids):
            boost = boosts.get(doc_id, 1.0)
            if result_provenance is not None and boost != 1.0:
                provenance = result_provenance.setdefault(
                    chunk_id, SearchResultProvenance()
                )
                provenance.community_boost = boost
            boosted.append((chunk_id, min(1.0, score * boost)))

        ranked = sorted(boosted, key=lambda x: x[1], reverse=True)
        return replace(context, candidates=ranked)
