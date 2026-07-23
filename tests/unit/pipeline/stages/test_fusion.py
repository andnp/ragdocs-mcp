from searchkernel.pipeline.stage import SearchContext
from searchkernel.pipeline.stages.fusion import FusionStage
from searchkernel.search.score_pipeline import ScorePipeline, ScorePipelineConfig


def _strategy_results():
    return {
        "semantic": [("chunk-1", 0.9), ("chunk-2", 0.7)],
        "keyword": [("chunk-2", 0.8), ("chunk-3", 0.5)],
    }


def test_fusion_stage_matches_score_pipeline_directly():
    strategy_results = _strategy_results()
    config = ScorePipelineConfig()

    expected = ScorePipeline(config).run(strategy_results)

    context = SearchContext(query="q", strategy_results=strategy_results)
    result = FusionStage(config).run(context)

    assert result.candidates == expected


def test_fusion_stage_does_not_mutate_input_context():
    strategy_results = _strategy_results()
    context = SearchContext(query="q", strategy_results=strategy_results)

    FusionStage().run(context)

    assert context.candidates == []


def test_fusion_stage_preserves_other_context_fields():
    context = SearchContext(
        query="q", strategy_results=_strategy_results(), metadata={"top_k": 10}
    )

    result = FusionStage().run(context)

    assert result.query == "q"
    assert result.metadata == {"top_k": 10}
