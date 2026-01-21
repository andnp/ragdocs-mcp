"""
Unit tests for ScorePipeline unified score processing.

Tests the consolidated pipeline that handles:
1. RRF fusion of multi-strategy results
2. Normalization to [0, 1] range
3. Sigmoid calibration for confidence interpretation
4. Optional time-based boosting
"""

from datetime import datetime, timedelta, timezone

import pytest

from src.search.score_pipeline import (
    ScorePipeline,
    ScorePipelineConfig,
    rrf_score,
)
from src.search.time_scoring import DecayConfig, TierConfig


# =============================================================================
# RRF Score Calculation
# =============================================================================


class TestRRFScore:
    """Tests for basic RRF score calculation."""

    def test_rrf_formula(self):
        """
        RRF formula is 1 / (k + rank).
        """
        assert rrf_score(0, 60.0) == pytest.approx(1 / 60)
        assert rrf_score(1, 60.0) == pytest.approx(1 / 61)
        assert rrf_score(5, 60.0) == pytest.approx(1 / 65)

    def test_rrf_decreases_with_rank(self):
        """
        Higher ranks produce lower scores.
        """
        scores = [rrf_score(i, 60.0) for i in range(5)]
        assert scores == sorted(scores, reverse=True)


# =============================================================================
# Pipeline Configuration
# =============================================================================


class TestScorePipelineConfig:
    """Tests for pipeline configuration defaults and customization."""

    def test_default_config(self):
        """
        Default config has sensible values for typical use.
        """
        config = ScorePipelineConfig()

        assert config.rrf_k == 60.0
        assert config.strategy_weights == {"semantic": 0.6, "keyword": 0.3, "graph": 0.1}
        assert config.calibration_threshold == 0.5
        assert config.calibration_steepness == 10.0
        assert config.time_scoring_mode is None

    def test_custom_config(self):
        """
        Custom config overrides defaults.
        """
        config = ScorePipelineConfig(
            rrf_k=30.0,
            strategy_weights={"semantic": 1.0, "keyword": 1.0},
            calibration_threshold=0.4,
            time_scoring_mode="tiers",
        )

        assert config.rrf_k == 30.0
        assert config.strategy_weights == {"semantic": 1.0, "keyword": 1.0}
        assert config.calibration_threshold == 0.4
        assert config.time_scoring_mode == "tiers"


# =============================================================================
# Fusion Stage
# =============================================================================


class TestFuseStage:
    """Tests for RRF fusion of multi-strategy results."""

    def test_fuse_empty_results(self):
        """
        Empty strategy results produce empty output.
        """
        pipeline = ScorePipeline()
        result = pipeline.fuse({})
        assert result == []

    def test_fuse_single_strategy(self):
        """
        Single strategy fusion preserves rank order with RRF scores.
        """
        pipeline = ScorePipeline(ScorePipelineConfig(
            strategy_weights={"semantic": 1.0}
        ))

        results = pipeline.fuse({
            "semantic": [("doc1", 0.9), ("doc2", 0.7), ("doc3", 0.5)]
        })

        doc_ids = [doc_id for doc_id, _ in results]
        assert doc_ids == ["doc1", "doc2", "doc3"]

        scores = [score for _, score in results]
        assert scores == sorted(scores, reverse=True)

    def test_fuse_multiple_strategies(self):
        """
        Documents appearing in multiple strategies get aggregated RRF scores.
        """
        pipeline = ScorePipeline(ScorePipelineConfig(
            strategy_weights={"semantic": 1.0, "keyword": 1.0}
        ))

        results = pipeline.fuse({
            "semantic": [("doc1", 0.9), ("doc2", 0.7)],
            "keyword": [("doc2", 0.8), ("doc3", 0.6)],
        })

        doc_ids = [doc_id for doc_id, _ in results]

        # doc2 appears in both lists, should rank highest
        assert doc_ids[0] == "doc2"

        # All docs should be present
        assert set(doc_ids) == {"doc1", "doc2", "doc3"}

    def test_fuse_with_weights(self):
        """
        Strategy weights affect relative ranking.
        """
        pipeline = ScorePipeline(ScorePipelineConfig(
            strategy_weights={"semantic": 2.0, "keyword": 1.0}
        ))

        results = pipeline.fuse({
            "semantic": [("doc1", 0.9)],
            "keyword": [("doc2", 0.9)],
        })

        # doc1 from semantic (2x weight) should rank above doc2 from keyword
        assert results[0][0] == "doc1"
        assert results[1][0] == "doc2"

        # Score ratio should reflect weight ratio
        assert results[0][1] / results[1][1] == pytest.approx(2.0)


# =============================================================================
# Normalize Stage
# =============================================================================


class TestNormalizeStage:
    """Tests for min-max normalization to [0, 1] range."""

    def test_normalize_empty_results(self):
        """
        Empty results produce empty output.
        """
        pipeline = ScorePipeline()
        result = pipeline.normalize([])
        assert result == []

    def test_normalize_single_result(self):
        """
        Single result normalizes to 1.0.
        """
        pipeline = ScorePipeline()
        result = pipeline.normalize([("doc1", 0.05)])

        assert len(result) == 1
        assert result[0][1] == 1.0

    def test_normalize_range(self):
        """
        Scores are scaled to [0, 1] with max→1.0 and min→0.0.
        """
        pipeline = ScorePipeline()

        fused = [("doc1", 0.08), ("doc2", 0.05), ("doc3", 0.02)]
        result = pipeline.normalize(fused)

        scores = [score for _, score in result]

        # Max should be 1.0, min should be 0.0
        assert max(scores) == 1.0
        assert min(scores) == 0.0

        # Order should be preserved
        assert result[0][0] == "doc1"
        assert result[2][0] == "doc3"

    def test_normalize_preserves_order(self):
        """
        Normalization preserves relative ranking.
        """
        pipeline = ScorePipeline()

        fused = [("a", 100.0), ("b", 50.0), ("c", 10.0)]
        result = pipeline.normalize(fused)

        doc_ids = [doc_id for doc_id, _ in result]
        assert doc_ids == ["a", "b", "c"]


# =============================================================================
# Calibrate Stage
# =============================================================================


class TestCalibrateStage:
    """Tests for sigmoid calibration."""

    def test_calibrate_empty_results(self):
        """
        Empty results produce empty output.
        """
        pipeline = ScorePipeline()
        result = pipeline.calibrate([])
        assert result == []

    def test_calibrate_threshold_gives_half(self):
        """
        Score at threshold produces ~0.5 confidence.
        """
        pipeline = ScorePipeline(ScorePipelineConfig(
            calibration_threshold=0.5,
            calibration_steepness=10.0,
        ))

        result = pipeline.calibrate([("doc1", 0.5)])

        assert result[0][1] == pytest.approx(0.5, abs=0.05)

    def test_calibrate_above_threshold(self):
        """
        Scores above threshold produce higher confidence.
        """
        pipeline = ScorePipeline(ScorePipelineConfig(
            calibration_threshold=0.5,
            calibration_steepness=10.0,
        ))

        result = pipeline.calibrate([("doc1", 0.8)])

        assert result[0][1] > 0.7

    def test_calibrate_below_threshold(self):
        """
        Scores below threshold produce lower confidence.
        """
        pipeline = ScorePipeline(ScorePipelineConfig(
            calibration_threshold=0.5,
            calibration_steepness=10.0,
        ))

        result = pipeline.calibrate([("doc1", 0.2)])

        assert result[0][1] < 0.3

    def test_calibrate_bounded_output(self):
        """
        All calibrated scores are in [0, 1].
        """
        pipeline = ScorePipeline()

        test_inputs = [("d1", 0.0), ("d2", 0.5), ("d3", 1.0), ("d4", 2.0)]
        result = pipeline.calibrate(test_inputs)

        for _, score in result:
            assert 0.0 <= score <= 1.0


# =============================================================================
# Boost Stage
# =============================================================================


class TestBoostStage:
    """Tests for time-based score boosting."""

    def test_boost_without_timestamps(self):
        """
        Missing timestamps returns original results.
        """
        pipeline = ScorePipeline(ScorePipelineConfig(
            time_scoring_mode="tiers"
        ))

        results = [("doc1", 0.8), ("doc2", 0.6)]
        boosted = pipeline.boost(results, {})

        assert boosted == results

    def test_boost_tier_recent_doc(self):
        """
        Recent documents get boost multiplier in tier mode.
        """
        config = ScorePipelineConfig(
            time_scoring_mode="tiers",
            time_scoring_config=TierConfig(
                recent_days=7,
                recent_boost=1.2,
                moderate_days=30,
                moderate_boost=1.1,
            ),
        )
        pipeline = ScorePipeline(config)

        now = datetime.now(timezone.utc)
        recent = now - timedelta(days=3)

        results = [("doc1", 0.5)]
        boosted = pipeline.boost(results, {"doc1": recent})

        # Should get 1.2x boost
        assert boosted[0][1] == pytest.approx(0.5 * 1.2)

    def test_boost_tier_old_doc(self):
        """
        Old documents get no boost in tier mode.
        """
        config = ScorePipelineConfig(
            time_scoring_mode="tiers",
            time_scoring_config=TierConfig(
                recent_days=7,
                recent_boost=1.2,
                moderate_days=30,
                moderate_boost=1.1,
            ),
        )
        pipeline = ScorePipeline(config)

        now = datetime.now(timezone.utc)
        old = now - timedelta(days=60)

        results = [("doc1", 0.5)]
        boosted = pipeline.boost(results, {"doc1": old})

        # No boost (1.0x)
        assert boosted[0][1] == pytest.approx(0.5)

    def test_boost_reorders_results(self):
        """
        Time boost can change ranking order.
        """
        config = ScorePipelineConfig(
            time_scoring_mode="tiers",
            time_scoring_config=TierConfig(
                recent_days=7,
                recent_boost=1.5,
                moderate_days=30,
                moderate_boost=1.0,
            ),
        )
        pipeline = ScorePipeline(config)

        now = datetime.now(timezone.utc)
        recent = now - timedelta(days=3)
        old = now - timedelta(days=60)

        # doc_old has higher base score but is old
        # doc_recent has lower base score but is recent
        results = [("doc_old", 0.6), ("doc_recent", 0.5)]
        timestamps = {"doc_old": old, "doc_recent": recent}

        boosted = pipeline.boost(results, timestamps)

        # doc_recent (0.5 * 1.5 = 0.75) > doc_old (0.6 * 1.0 = 0.6)
        assert boosted[0][0] == "doc_recent"
        assert boosted[1][0] == "doc_old"

    def test_boost_decay_mode(self):
        """
        Decay mode applies exponential time decay.
        """
        config = ScorePipelineConfig(
            time_scoring_mode="decay",
            time_scoring_config=DecayConfig(
                half_life_days=7.0,
                min_score=0.1,
            ),
        )
        pipeline = ScorePipeline(config)

        now = datetime.now(timezone.utc)
        week_old = now - timedelta(days=7)

        results = [("doc1", 1.0)]
        boosted = pipeline.boost(results, {"doc1": week_old})

        # After half-life, score should be ~0.5x
        assert boosted[0][1] == pytest.approx(0.5, abs=0.05)


# =============================================================================
# Full Pipeline Integration
# =============================================================================


class TestFullPipeline:
    """Integration tests for complete pipeline execution."""

    def test_run_basic(self):
        """
        Basic pipeline run produces normalized, calibrated scores.
        """
        pipeline = ScorePipeline(ScorePipelineConfig(
            strategy_weights={"semantic": 1.0, "keyword": 1.0},
            calibration_threshold=0.5,
            calibration_steepness=10.0,
        ))

        strategy_results = {
            "semantic": [("doc1", 0.9), ("doc2", 0.7), ("doc3", 0.5)],
            "keyword": [("doc2", 0.8), ("doc4", 0.6)],
        }

        results = pipeline.run(strategy_results)

        # All scores in [0, 1]
        for _, score in results:
            assert 0.0 <= score <= 1.0

        # Results are sorted by score descending
        scores = [score for _, score in results]
        assert scores == sorted(scores, reverse=True)

    def test_run_with_time_boost(self):
        """
        Pipeline with time boosting applies all stages.
        """
        config = ScorePipelineConfig(
            strategy_weights={"semantic": 1.0},
            calibration_threshold=0.5,
            calibration_steepness=5.0,
            time_scoring_mode="tiers",
            time_scoring_config=TierConfig(
                recent_days=7,
                recent_boost=1.2,
            ),
        )
        pipeline = ScorePipeline(config)

        strategy_results = {
            "semantic": [("doc1", 0.9), ("doc2", 0.7)],
        }

        now = datetime.now(timezone.utc)
        timestamps = {"doc1": now - timedelta(days=3)}

        results = pipeline.run(strategy_results, timestamps)

        # All scores still in valid range (boost may push > 1.0 but still reasonable)
        for _, score in results:
            assert 0.0 <= score <= 1.5  # Allow some boost headroom

    def test_run_empty_input(self):
        """
        Empty strategy results produce empty output.
        """
        pipeline = ScorePipeline()
        results = pipeline.run({})
        assert results == []

    def test_run_single_doc(self):
        """
        Single document through pipeline produces valid output.
        """
        pipeline = ScorePipeline(ScorePipelineConfig(
            strategy_weights={"semantic": 1.0},
            calibration_threshold=0.5,
            calibration_steepness=10.0,
        ))

        results = pipeline.run({"semantic": [("doc1", 0.9)]})

        assert len(results) == 1
        assert results[0][0] == "doc1"
        # Single normalized score is 1.0, which is above 0.5 threshold → high confidence
        assert results[0][1] > 0.5


# =============================================================================
# Stage Isolation
# =============================================================================


class TestStageIsolation:
    """Tests verifying each stage can be called independently."""

    def test_stages_callable_independently(self):
        """
        Each stage method works in isolation.
        """
        pipeline = ScorePipeline()

        # Fuse
        fused = pipeline.fuse({"semantic": [("doc1", 0.9)]})
        assert len(fused) == 1

        # Normalize
        normalized = pipeline.normalize([("doc1", 0.08), ("doc2", 0.02)])
        assert normalized[0][1] == 1.0
        assert normalized[1][1] == 0.0

        # Calibrate
        calibrated = pipeline.calibrate([("doc1", 0.5)])
        assert 0.0 <= calibrated[0][1] <= 1.0

        # Boost (with timestamps)
        config = ScorePipelineConfig(
            time_scoring_mode="tiers",
            time_scoring_config=TierConfig(),
        )
        pipeline_with_boost = ScorePipeline(config)
        now = datetime.now(timezone.utc)
        boosted = pipeline_with_boost.boost(
            [("doc1", 0.5)],
            {"doc1": now - timedelta(days=3)},
        )
        assert boosted[0][1] >= 0.5  # Should get boost
