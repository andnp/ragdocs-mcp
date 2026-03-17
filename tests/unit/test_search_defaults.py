import pytest

from src.config import SearchConfig
from src.search.calibration import calibrate_score


class TestSearchDefaults:
    def test_search_config_defaults(self):
        config = SearchConfig()

        assert config.min_confidence == 0.3
        assert config.semantic_weight == 1.0
        assert config.keyword_weight == 1.0

    def test_calibration_default_midpoint(self):
        score = calibrate_score(0.035)

        assert score == pytest.approx(0.5, abs=0.01)
