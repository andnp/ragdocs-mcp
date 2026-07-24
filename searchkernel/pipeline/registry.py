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
from searchkernel.pipeline.stages.community_boost import (
    BoostByCommunity,
    CommunityBoostStage,
)
from searchkernel.pipeline.stages.dedup_rerank import DedupRerankStage
from searchkernel.pipeline.stages.effective_top_k import EffectiveTopKStage
from searchkernel.pipeline.stages.fusion import FusionStage
from searchkernel.pipeline.stages.graph_expand import (
    BuildChunkCandidates,
    GraphExpandStage,
    RankNeighbors,
)
from searchkernel.pipeline.stages.hydrate import HydrateChunkResult, HydrateStage
from searchkernel.pipeline.stages.parent_expansion import ParentExpansionStage
from searchkernel.pipeline.stages.project_filter import ProjectFilterStage
from searchkernel.pipeline.stages.project_uplift import GetChunk, ProjectUpliftStage
from searchkernel.pipeline.stages.provenance import ProvenanceStage
from searchkernel.pipeline.stages.rag_fusion import (
    GenerateQueryVariants,
    RAGFusionConfig,
    RAGFusionStage,
)
from searchkernel.pipeline.stages.retrieve import RetrieveStage, Searcher
from searchkernel.pipeline.stages.routing import RoutingStage
from searchkernel.pipeline.stages.seed_bookkeeping import SeedBookkeepingStage
from searchkernel.pipeline.stages.source_filter import SourceFilterStage
from searchkernel.pipeline.stages.strategy_results import StrategyResultsStage
from searchkernel.pipeline.stages.tag_expansion import (
    ExpandQueryWithTags,
    TagExpansionStage,
)
from searchkernel.search.pipeline import SearchPipelineConfig
from searchkernel.search.score_pipeline import ScorePipelineConfig


@dataclass(frozen=True)
class StageDeps:
    """Orchestrator-bound callables a registry factory may need."""

    search_vector: Searcher | None = None
    search_keyword: Searcher | None = None
    rank_neighbors: RankNeighbors | None = None
    build_chunk_candidates: BuildChunkCandidates | None = None
    generate_query_variants: GenerateQueryVariants | None = None
    rag_fusion_retrieve: Searcher | None = None
    hydrate_chunk_result: HydrateChunkResult | None = None
    expand_query_with_tags: ExpandQueryWithTags | None = None
    boost_by_community: BoostByCommunity | None = None
    get_chunk: GetChunk | None = None
    get_parent_chunk: GetChunk | None = None


StageFactory = Callable[[dict[str, Any], StageDeps], "SearchStage | AsyncSearchStage"]


def _make_routing(config: dict[str, Any], deps: StageDeps) -> SearchStage:
    return RoutingStage()


def _make_effective_top_k(config: dict[str, Any], deps: StageDeps) -> SearchStage:
    return EffectiveTopKStage()


def _make_retrieve(config: dict[str, Any], deps: StageDeps) -> AsyncSearchStage:
    return RetrieveStage(deps.search_vector, deps.search_keyword)


def _make_seed_bookkeeping(config: dict[str, Any], deps: StageDeps) -> SearchStage:
    return SeedBookkeepingStage()


def _make_graph_expand(config: dict[str, Any], deps: StageDeps) -> SearchStage:
    return GraphExpandStage(deps.rank_neighbors, deps.build_chunk_candidates)


def _make_strategy_results(config: dict[str, Any], deps: StageDeps) -> SearchStage:
    return StrategyResultsStage()


def _make_fusion(config: dict[str, Any], deps: StageDeps) -> SearchStage:
    return FusionStage(ScorePipelineConfig(**config))


def _make_dedup_rerank(config: dict[str, Any], deps: StageDeps) -> SearchStage:
    return DedupRerankStage(SearchPipelineConfig(**config))


def _make_rag_fusion(config: dict[str, Any], deps: StageDeps) -> AsyncSearchStage:
    return RAGFusionStage(
        deps.generate_query_variants,
        deps.rag_fusion_retrieve,
        RAGFusionConfig(**config),
    )


def _make_provenance(config: dict[str, Any], deps: StageDeps) -> SearchStage:
    return ProvenanceStage()


def _make_hydrate(config: dict[str, Any], deps: StageDeps) -> SearchStage:
    return HydrateStage(deps.hydrate_chunk_result)


def _make_tag_expansion(config: dict[str, Any], deps: StageDeps) -> SearchStage:
    return TagExpansionStage(deps.expand_query_with_tags)


def _make_community_boost(config: dict[str, Any], deps: StageDeps) -> SearchStage:
    return CommunityBoostStage(deps.boost_by_community)


def _make_project_uplift(config: dict[str, Any], deps: StageDeps) -> SearchStage:
    return ProjectUpliftStage(deps.get_chunk, config.get("uplift", 1.2))


def _make_project_filter(config: dict[str, Any], deps: StageDeps) -> SearchStage:
    return ProjectFilterStage(deps.get_chunk)


def _make_source_filter(config: dict[str, Any], deps: StageDeps) -> SearchStage:
    return SourceFilterStage(deps.get_chunk)


def _make_parent_expansion(config: dict[str, Any], deps: StageDeps) -> SearchStage:
    return ParentExpansionStage(deps.get_chunk, deps.get_parent_chunk)


DEFAULT_QUERY_STAGE_REGISTRY: dict[str, StageFactory] = {
    "routing": _make_routing,
    "effective_top_k": _make_effective_top_k,
    "retrieve": _make_retrieve,
    "seed_bookkeeping": _make_seed_bookkeeping,
    "graph_expand": _make_graph_expand,
    "strategy_results": _make_strategy_results,
    "fusion": _make_fusion,
    "dedup_rerank": _make_dedup_rerank,
    "rag_fusion": _make_rag_fusion,
    "provenance": _make_provenance,
    "hydrate": _make_hydrate,
    "tag_expansion": _make_tag_expansion,
    "community_boost": _make_community_boost,
    "project_uplift": _make_project_uplift,
    "project_filter": _make_project_filter,
    "source_filter": _make_source_filter,
    "parent_expansion": _make_parent_expansion,
}
