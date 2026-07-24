from searchkernel.pipeline.stage import SearchContext
from searchkernel.pipeline.stages.calibrate import CalibrateStage
from searchkernel.search.calibration import calibrate_results


def _candidates():
    return [("chunk_a", 0.09), ("chunk_b", 0.05), ("chunk_c", 0.01)]


def test_calibrate_stage_matches_score_pipeline_calibrate_directly():
    expected = calibrate_results(_candidates(), threshold=0.02, steepness=150.0)

    context = SearchContext(query="q", candidates=_candidates())
    result = CalibrateStage().run(context)

    assert result.candidates == expected


def test_calibrate_stage_empty_candidates():
    context = SearchContext(query="q", candidates=[])
    result = CalibrateStage().run(context)

    assert result.candidates == []


def test_calibrate_stage_does_not_mutate_input_context():
    context = SearchContext(query="q", candidates=_candidates())

    CalibrateStage().run(context)

    assert context.candidates == _candidates()
