from searchkernel.pipeline.stage import SearchContext
from searchkernel.pipeline.stages.threshold import ThresholdStage
from searchkernel.search.filters import filter_by_confidence


def _candidates():
    return [("chunk_a", 0.9), ("chunk_b", 0.5), ("chunk_c", 0.1)]


def test_threshold_stage_matches_filter_by_confidence_directly():
    expected = filter_by_confidence(_candidates(), 0.5)

    context = SearchContext(query="q", candidates=_candidates())
    result = ThresholdStage(0.5).run(context)

    assert result.candidates == expected


def test_threshold_stage_zero_confidence_keeps_all():
    context = SearchContext(query="q", candidates=_candidates())
    result = ThresholdStage(0.0).run(context)

    assert result.candidates == _candidates()


def test_threshold_stage_does_not_mutate_input_context():
    context = SearchContext(query="q", candidates=_candidates())

    ThresholdStage(0.5).run(context)

    assert context.candidates == _candidates()
