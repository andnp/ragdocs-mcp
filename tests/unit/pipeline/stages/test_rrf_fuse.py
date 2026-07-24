from searchkernel.pipeline.stage import SearchContext
from searchkernel.pipeline.stages.rrf_fuse import RRFFuseStage
from searchkernel.search.score_pipeline import ScorePipeline, ScorePipelineConfig


def _strategy_results():
    return {
        "semantic": [("chunk-1", 0.9), ("chunk-2", 0.7)],
        "keyword": [("chunk-2", 0.8), ("chunk-3", 0.5)],
    }


def test_rrf_fuse_stage_matches_score_pipeline_fuse_directly():
    config = ScorePipelineConfig()
    expected = ScorePipeline(config).fuse(_strategy_results())

    context = SearchContext(query="q", strategy_results=_strategy_results())
    result = RRFFuseStage(config).run(context)

    assert result.candidates == expected


def test_rrf_fuse_stage_does_not_mutate_input_context():
    context = SearchContext(query="q", strategy_results=_strategy_results())

    RRFFuseStage().run(context)

    assert context.candidates == []


def test_rrf_fuse_stage_preserves_other_context_fields():
    context = SearchContext(
        query="q", strategy_results=_strategy_results(), metadata={"top_k": 10}
    )

    result = RRFFuseStage().run(context)

    assert result.query == "q"
    assert result.metadata == {"top_k": 10}
