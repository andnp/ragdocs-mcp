import pytest

from src.config import SearchConfig
from src.memory.search import apply_memory_decay
from src.search.calibration import calibrate_score


class TestSearchDefaults:
    def test_search_config_defaults(self):
        """
        Verify default search thresholds and weights.

        Ensures configured defaults align with documented calibration behavior.
        """
        config = SearchConfig()

        assert config.min_confidence == 0.3
        assert config.score_calibration_threshold == 0.035
        assert config.semantic_weight == 1.0
        assert config.keyword_weight == 1.0

    def test_calibration_default_midpoint(self):
        """
        Confirm default calibration threshold maps to midpoint.

        The default threshold should produce ~0.5 confidence.
        """
        score = calibrate_score(0.035)

        assert score == pytest.approx(0.5, abs=0.01)

    def test_apply_memory_decay_no_penalty_when_missing_timestamp(self):
        """
        Ensure no penalty is applied for missing timestamps.

        For backward compatibility with legacy memories that lack created_at,
        the decay function returns a 1.0 multiplier (no penalty).
        """
        decayed = apply_memory_decay(0.5, None, 0.9, 0.1)

        assert decayed == pytest.approx(0.5)
