"""Stage registry: maps a `StageSpec.name` to a concrete stage factory.

Stages that only need config (`RoutingStage`, `FusionStage`,
`DedupRerankStage`) ignore `StageDeps`; stages parameterized over
orchestrator-bound callables (`RetrieveStage`, `GraphExpandStage`) pull
them from it. Keeping the registry itself free of `SearchOrchestrator`
imports is what lets `PipelineExecutor` resolve "which stage runs here"
from data (a `PipelineSpec`) instead of hardcoded construction --
registering a new stage, or swapping one slot's implementation, becomes
an edit to the registry + spec rather than to the orchestrator.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from searchkernel.pipeline.stage import AsyncSearchStage, SearchStage
from searchkernel.pipeline.stages.dedup_rerank import DedupRerankStage
from searchkernel.pipeline.stages.fusion import FusionStage
from searchkernel.pipeline.stages.graph_expand import (
    BuildChunkCandidates,
    GraphExpandStage,
    RankNeighbors,
)
from searchkernel.pipeline.stages.retrieve import RetrieveStage, Searcher
from searchkernel.pipeline.stages.routing import RoutingStage
from searchkernel.search.pipeline import SearchPipelineConfig
from searchkernel.search.score_pipeline import ScorePipelineConfig


@dataclass(frozen=True)
class StageDeps:
    """Orchestrator-bound callables a registry factory may need."""

    search_vector: Searcher | None = None
    search_keyword: Searcher | None = None
    rank_neighbors: RankNeighbors | None = None
    build_chunk_candidates: BuildChunkCandidates | None = None


StageFactory = Callable[[dict[str, Any], StageDeps], "SearchStage | AsyncSearchStage"]


def _make_routing(config: dict[str, Any], deps: StageDeps) -> SearchStage:
    return RoutingStage()


def _make_retrieve(config: dict[str, Any], deps: StageDeps) -> AsyncSearchStage:
    return RetrieveStage(deps.search_vector, deps.search_keyword)


def _make_graph_expand(config: dict[str, Any], deps: StageDeps) -> SearchStage:
    return GraphExpandStage(deps.rank_neighbors, deps.build_chunk_candidates)


def _make_fusion(config: dict[str, Any], deps: StageDeps) -> SearchStage:
    return FusionStage(ScorePipelineConfig(**config))


def _make_dedup_rerank(config: dict[str, Any], deps: StageDeps) -> SearchStage:
    return DedupRerankStage(SearchPipelineConfig(**config))


DEFAULT_QUERY_STAGE_REGISTRY: dict[str, StageFactory] = {
    "routing": _make_routing,
    "retrieve": _make_retrieve,
    "graph_expand": _make_graph_expand,
    "fusion": _make_fusion,
    "dedup_rerank": _make_dedup_rerank,
}
