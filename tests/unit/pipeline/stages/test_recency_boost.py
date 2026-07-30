from datetime import UTC, datetime, timedelta

import pytest

from searchkernel.pipeline.stage import SearchContext
from searchkernel.pipeline.stages.recency_boost import RecencyBoostStage
from searchkernel.search.score_pipeline import ScorePipeline, ScorePipelineConfig
from searchkernel.search.time_scoring import DecayConfig


def _candidates():
    return [("chunk_a", 0.9), ("chunk_b", 0.5)]


def _timestamps():
    now = datetime.now(UTC)
    return {
        "chunk_a": now - timedelta(days=1),
        "chunk_b": now - timedelta(days=90),
    }


def test_recency_boost_stage_matches_score_pipeline_boost_directly():
    config = ScorePipelineConfig(
        time_scoring_mode="decay", time_scoring_config=DecayConfig()
    )
    expected = ScorePipeline(config).boost(_candidates(), _timestamps())

    context = SearchContext(query="q", candidates=_candidates())
    result = RecencyBoostStage(config).run(context, _timestamps())

    # boost()'s reference_time defaults to datetime.now(), so the two
    # calls' outputs can differ by floating-point noise on the decay
    # curve; compare ids/order and scores within tolerance instead.
    assert [chunk_id for chunk_id, _ in result.candidates] == [
        chunk_id for chunk_id, _ in expected
    ]
    for (_, actual_score), (_, expected_score) in zip(result.candidates, expected):
        assert actual_score == pytest.approx(expected_score)


def test_recency_boost_stage_noop_without_timestamps():
    context = SearchContext(query="q", candidates=_candidates())

    result = RecencyBoostStage().run(context, None)

    assert result.candidates == _candidates()


def test_recency_boost_stage_noop_without_configured_mode():
    context = SearchContext(query="q", candidates=_candidates())

    result = RecencyBoostStage().run(context, _timestamps())

    assert result.candidates == _candidates()


def test_recency_boost_stage_does_not_mutate_input_context():
    config = ScorePipelineConfig(
        time_scoring_mode="decay", time_scoring_config=DecayConfig()
    )
    context = SearchContext(query="q", candidates=_candidates())

    RecencyBoostStage(config).run(context, _timestamps())

    assert context.candidates == _candidates()
