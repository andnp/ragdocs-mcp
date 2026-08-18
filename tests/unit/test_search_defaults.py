from mcp_markdown_ragdocs.app.search import to_record_search_config
from mcp_markdown_ragdocs.config import SearchConfig


class TestSearchDefaults:
    def test_search_config_defaults(self):
        config = SearchConfig()

        assert config.min_confidence == 0.3
        assert config.max_chunks_per_doc == 1
        assert config.abstention_threshold is None
        assert config.semantic_weight == 1.0
        assert config.keyword_weight == 1.0

    def test_abstention_threshold_maps_to_raw_kernel_score_floor(self):
        config = SearchConfig(abstention_threshold=0.025)

        assert to_record_search_config(config).minimum_score == 0.025
