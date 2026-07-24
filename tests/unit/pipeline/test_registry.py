from searchkernel.pipeline.registry import DEFAULT_QUERY_STAGE_REGISTRY, StageDeps
from searchkernel.pipeline.stage import AsyncSearchStage, SearchStage
from searchkernel.pipeline.stages.dedup_rerank import DedupRerankStage
from searchkernel.pipeline.stages.fusion import FusionStage
from searchkernel.pipeline.stages.graph_expand import GraphExpandStage
from searchkernel.pipeline.stages.hydrate import HydrateStage
from searchkernel.pipeline.stages.provenance import ProvenanceStage
from searchkernel.pipeline.stages.rag_fusion import RAGFusionStage
from searchkernel.pipeline.stages.retrieve import RetrieveStage
from searchkernel.pipeline.stages.routing import RoutingStage
from searchkernel.pipeline.stages.tag_expansion import TagExpansionStage


def test_registry_has_the_five_default_query_stages_plus_rag_fusion():
    assert set(DEFAULT_QUERY_STAGE_REGISTRY) == {
        "routing",
        "retrieve",
        "graph_expand",
        "fusion",
        "dedup_rerank",
        "rag_fusion",
        "provenance",
        "hydrate",
        "tag_expansion",
    }


def test_routing_factory_needs_no_deps():
    stage = DEFAULT_QUERY_STAGE_REGISTRY["routing"]({}, StageDeps())

    assert isinstance(stage, RoutingStage)
    assert isinstance(stage, SearchStage)


def test_retrieve_factory_injects_searchers_from_deps():
    async def search_vector(*_args):
        return []

    async def search_keyword(*_args):
        return []

    deps = StageDeps(search_vector=search_vector, search_keyword=search_keyword)
    stage = DEFAULT_QUERY_STAGE_REGISTRY["retrieve"]({}, deps)

    assert isinstance(stage, RetrieveStage)
    assert isinstance(stage, AsyncSearchStage)
    assert stage._search_vector is search_vector
    assert stage._search_keyword is search_keyword


def test_graph_expand_factory_injects_callables_from_deps():
    def rank_neighbors(_seed_scores):
        return []

    def build_chunk_candidates(_neighbor_doc_ids, _top_k, _excluded):
        return []

    deps = StageDeps(
        rank_neighbors=rank_neighbors, build_chunk_candidates=build_chunk_candidates
    )
    stage = DEFAULT_QUERY_STAGE_REGISTRY["graph_expand"]({}, deps)

    assert isinstance(stage, GraphExpandStage)
    assert stage._rank_neighbors is rank_neighbors
    assert stage._build_chunk_candidates is build_chunk_candidates


def test_fusion_factory_builds_config_from_dict():
    stage = DEFAULT_QUERY_STAGE_REGISTRY["fusion"](
        {"strategy_weights": {"semantic": 1.0, "keyword": 0.0, "graph": 0.0}},
        StageDeps(),
    )

    assert isinstance(stage, FusionStage)


def test_dedup_rerank_factory_builds_config_from_dict():
    stage = DEFAULT_QUERY_STAGE_REGISTRY["dedup_rerank"](
        {"min_confidence": 0.5, "reranking_enabled": False}, StageDeps()
    )

    assert isinstance(stage, DedupRerankStage)


def test_rag_fusion_factory_injects_callables_and_config_from_deps():
    async def generate_query_variants(_query, _num_variants):
        return []

    async def rag_fusion_retrieve(*_args):
        return []

    deps = StageDeps(
        generate_query_variants=generate_query_variants,
        rag_fusion_retrieve=rag_fusion_retrieve,
    )
    stage = DEFAULT_QUERY_STAGE_REGISTRY["rag_fusion"]({"enabled": True}, deps)

    assert isinstance(stage, RAGFusionStage)
    assert isinstance(stage, AsyncSearchStage)
    assert stage._generate_query_variants is generate_query_variants
    assert stage._retrieve is rag_fusion_retrieve
    assert stage._config.enabled is True


def test_provenance_factory_needs_no_deps():
    stage = DEFAULT_QUERY_STAGE_REGISTRY["provenance"]({}, StageDeps())

    assert isinstance(stage, ProvenanceStage)
    assert isinstance(stage, SearchStage)


def test_hydrate_factory_injects_hydrate_chunk_result_from_deps():
    def hydrate_chunk_result(_chunk_id, _score):
        return None

    deps = StageDeps(hydrate_chunk_result=hydrate_chunk_result)
    stage = DEFAULT_QUERY_STAGE_REGISTRY["hydrate"]({}, deps)

    assert isinstance(stage, HydrateStage)
    assert stage._hydrate_chunk_result is hydrate_chunk_result


def test_tag_expansion_factory_injects_expand_query_with_tags_from_deps():
    def expand_query_with_tags(_initial_results, _top_k):
        return []

    deps = StageDeps(expand_query_with_tags=expand_query_with_tags)
    stage = DEFAULT_QUERY_STAGE_REGISTRY["tag_expansion"]({}, deps)

    assert isinstance(stage, TagExpansionStage)
    assert stage._expand_query_with_tags is expand_query_with_tags
