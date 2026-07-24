"""W4a acceptance test: DEFAULT_QUERY_SPEC-via-PipelineExecutor eval parity.

Proves the extraction claim behind W4a's query toolkit -- that
DEFAULT_QUERY_SPEC + DEFAULT_QUERY_STAGE_REGISTRY are a complete,
standalone description of the query pipeline -- by driving the spec
through PipelineExecutor independently of SearchOrchestrator.query() and
asserting the two produce IDENTICAL results (same doc/chunk ordering,
same scores) on the WE eval harness's golden fixture corpus.

SearchOrchestrator.query() itself walks this same spec since the W4a
query()-rewrite pass, so this is not a comparison against a distinct
implementation; it independently re-walks DEFAULT_QUERY_SPEC via
PipelineExecutor.run_stage from test code -- exercising the registry's
factories and StageDeps wiring exactly as any composition root other than
SearchOrchestrator would have to -- rather than re-invoking query() and
trivially comparing it to itself.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from searchkernel.pipeline.default_query_spec import DEFAULT_QUERY_SPEC
from searchkernel.pipeline.executor import PipelineExecutor
from searchkernel.pipeline.registry import DEFAULT_QUERY_STAGE_REGISTRY, StageDeps
from searchkernel.pipeline.stage import SearchContext
from searchkernel.search.orchestrator import SearchOrchestrator
from tests.search.evaluation_harness import (
    SEARCH_EVALUATION_CASES,
    SearchEvaluationCase,
    build_search_evaluation_harness,
)


@pytest.fixture(scope="module")
def search_evaluation_harness(shared_embedding_model, tmp_path_factory):
    return build_search_evaluation_harness(
        tmp_path_factory.mktemp("default-spec-parity"),
        shared_embedding_model,
    )


async def _run_default_spec_via_executor(
    orchestrator: SearchOrchestrator,
    case: SearchEvaluationCase,
    top_k: int,
) -> SearchContext:
    """Walk DEFAULT_QUERY_SPEC through a fresh PipelineExecutor, not query().

    Builds the same initial SearchContext/StageDeps orchestrator.query()
    builds, then drives DEFAULT_QUERY_SPEC.stages through a standalone
    PipelineExecutor -- the same registry, the same stage classes, but a
    walk owned by this test rather than by SearchOrchestrator.
    """
    query_context = orchestrator._create_query_execution_context()
    executor = PipelineExecutor(DEFAULT_QUERY_STAGE_REGISTRY)
    deps = StageDeps(
        search_vector=orchestrator._search_vector,
        search_keyword=orchestrator._search_keyword,
        rank_neighbors=orchestrator._get_ranked_graph_neighbors,
        build_chunk_candidates=orchestrator._build_graph_chunk_candidates,
        expand_query_with_tags=orchestrator._run_tag_expansion,
        boost_by_community=orchestrator._graph.boost_by_community,
        get_chunk=query_context.get_vector_chunk,
        get_parent_chunk=query_context.get_parent_chunk,
        hydrate_chunk_result=query_context.hydrate_chunk_result,
    )
    project_filter = list(case.project_filter) if case.project_filter is not None else None
    context = SearchContext(
        query=case.query,
        metadata={
            "base_semantic_weight": orchestrator._config.search.semantic_weight,
            "base_keyword_weight": orchestrator._config.search.keyword_weight,
            "base_graph_weight": 1.0,
            "requested_top_k": top_k,
            "top_n": case.top_n,
            "project_filter": project_filter,
            "excluded_files": None,
            "docs_root": orchestrator.documents_path,
            "source_filter": None,
            "active_project": case.project_context or orchestrator._config.detected_project,
        },
    )

    for stage_spec in DEFAULT_QUERY_SPEC.stages:
        name = stage_spec.name
        if name == "dedup_rerank":
            pipeline = orchestrator._resolve_pipeline(
                None, disable_reranking=context.metadata["skip_tag_expansion"]
            )
            dedup_metadata = dict(context.metadata)
            dedup_metadata["get_embedding"] = query_context.get_chunk_embedding
            dedup_metadata["get_content"] = query_context.get_chunk_content
            dedup_metadata["top_n"] = case.top_n
            context = pipeline.run(replace(context, metadata=dedup_metadata))
            continue

        config = dict(stage_spec.config)
        if name == "fusion":
            config["strategy_weights"] = context.metadata["strategy_weights"]
        context = await executor.run_stage(name, config, context, deps)

    return context


@pytest.mark.asyncio
@pytest.mark.parametrize("case", SEARCH_EVALUATION_CASES, ids=lambda case: case.case_id)
async def test_default_spec_matches_legacy_orchestrator(
    search_evaluation_harness, case: SearchEvaluationCase
) -> None:
    orchestrator = search_evaluation_harness.orchestrator
    top_k = max(10, case.top_n * 2)
    project_filter = list(case.project_filter) if case.project_filter is not None else None

    legacy_results, legacy_compression_stats, legacy_strategy_stats = await orchestrator.query(
        case.query,
        top_k=top_k,
        top_n=case.top_n,
        project_context=case.project_context,
        project_filter=project_filter,
    )

    spec_context = await _run_default_spec_via_executor(orchestrator, case, top_k)
    spec_results = spec_context.metadata["chunk_results"]
    spec_compression_stats = spec_context.metadata["compression_stats"]

    assert [(r.chunk_id, r.doc_id, r.score) for r in spec_results] == [
        (r.chunk_id, r.doc_id, r.score) for r in legacy_results
    ]
    assert [r.content for r in spec_results] == [r.content for r in legacy_results]
    assert spec_compression_stats == legacy_compression_stats
    assert len(spec_context.metadata["vector_results"]) == legacy_strategy_stats.vector_count
