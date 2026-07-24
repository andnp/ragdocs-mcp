from searchkernel.pipeline.default_query_spec import DEFAULT_QUERY_SPEC
from searchkernel.pipeline.registry import DEFAULT_QUERY_STAGE_REGISTRY


def test_default_query_spec_matches_the_orchestrators_hand_wired_order():
    assert DEFAULT_QUERY_SPEC.stage_names() == (
        "routing",
        "effective_top_k",
        "retrieve",
        "seed_bookkeeping",
        "tag_expansion",
        "graph_expand",
        "strategy_results",
        "provenance",
        "fusion",
        "community_boost",
        "project_uplift",
        "project_filter",
        "source_filter",
        "dedup_rerank",
        "parent_expansion",
        "hydrate",
    )


def test_default_query_spec_stage_names_are_all_registered():
    assert set(DEFAULT_QUERY_SPEC.stage_names()) <= set(DEFAULT_QUERY_STAGE_REGISTRY)
