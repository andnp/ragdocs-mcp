"""GraphExpandStage: graph one-hop expansion query stage.

Lifted from SearchOrchestrator.query's rank-neighbors +
build_graph_chunk_candidates sequence. Parameterized over the two
callables (not a concrete GraphStore/VectorIndex) so it composes over the
orchestrator's own _get_ranked_graph_neighbors/
_build_graph_chunk_candidates -- staying monkeypatchable in existing
tests -- once wired in.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

from searchkernel.pipeline.stage import SearchContext

RankNeighbors = Callable[[dict[str, float]], list[tuple[str, float]]]
BuildChunkCandidates = Callable[[list[str], int, "set[str] | None"], list[str]]

_SEED_SCORES_KEY = "seed_scores"
_TOP_K_KEY = "top_k"
_EXCLUDED_CHUNK_IDS_KEY = "excluded_chunk_ids"
_GRAPH_CHUNK_IDS_KEY = "graph_chunk_ids"
_GRAPH_DOC_SCORES_KEY = "graph_doc_scores"


class GraphExpandStage:
    """One-hop graph expansion from vector/keyword seed scores.

    Expects `context.metadata` to carry `seed_scores` (dict[str, float]),
    `top_k` (int) and `excluded_chunk_ids` (set[str] | None). Writes
    `context.metadata["graph_chunk_ids"]` (list[str]) and
    `["graph_doc_scores"]` (dict[str, float]).
    """

    name = "graph_expand"

    def __init__(
        self,
        rank_neighbors: RankNeighbors,
        build_chunk_candidates: BuildChunkCandidates,
    ):
        self._rank_neighbors = rank_neighbors
        self._build_chunk_candidates = build_chunk_candidates

    def run(self, context: SearchContext) -> SearchContext:
        seed_scores = context.metadata[_SEED_SCORES_KEY]
        top_k = context.metadata[_TOP_K_KEY]
        excluded_chunk_ids = context.metadata.get(_EXCLUDED_CHUNK_IDS_KEY)

        ranked_neighbors = self._rank_neighbors(seed_scores)
        neighbor_doc_ids = [doc_id for doc_id, _score in ranked_neighbors]
        doc_scores = {doc_id: score for doc_id, score in ranked_neighbors}
        chunk_ids = self._build_chunk_candidates(
            neighbor_doc_ids, top_k, excluded_chunk_ids
        )

        metadata = dict(context.metadata)
        metadata[_GRAPH_CHUNK_IDS_KEY] = chunk_ids
        metadata[_GRAPH_DOC_SCORES_KEY] = doc_scores
        return replace(context, metadata=metadata)
