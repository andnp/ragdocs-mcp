import pytest

from src.config import SearchConfig
from src.memory.search import apply_recency_boost
from src.search.calibration import calibrate_score


class TestSearchDefaults:
    def test_search_config_defaults(self):
        config = SearchConfig()

        assert config.min_confidence == 0.3
        assert config.score_calibration_threshold == 0.035
        assert config.semantic_weight == 1.0
        assert config.keyword_weight == 1.0

    def test_calibration_default_midpoint(self):
        score = calibrate_score(0.035)

        assert score == pytest.approx(0.5, abs=0.01)

    def test_apply_recency_boost_returns_base_score_when_missing_timestamp(self):
        boosted = apply_recency_boost(
            score=0.5,
            created_at=None,
            boost_window_days=14,
            max_boost_amount=0.2,
            boost_decay_rate=0.95,
        )

        assert boosted == pytest.approx(0.5)
