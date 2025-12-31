import pytest

from src.search.classifier import (
    QueryType,
    classify_query,
    get_adaptive_weights,
)


class TestClassifyQuery:
    def test_factual_camel_case(self):
        assert classify_query("getUserById function") == QueryType.FACTUAL

    def test_factual_snake_case(self):
        assert classify_query("get_user_by_id") == QueryType.FACTUAL

    def test_factual_backticks(self):
        assert classify_query("how to use `asyncio`") == QueryType.FACTUAL

    def test_factual_version_number(self):
        assert classify_query("changes in v2.0.1") == QueryType.FACTUAL

    def test_factual_quoted_phrase(self):
        assert classify_query('error "module not found"') == QueryType.FACTUAL

    def test_navigational_section_keyword(self):
        assert classify_query("installation section") == QueryType.NAVIGATIONAL

    def test_navigational_guide_keyword(self):
        assert classify_query("quickstart guide") == QueryType.NAVIGATIONAL

    def test_navigational_in_the_phrase(self):
        assert classify_query("options in the config") == QueryType.NAVIGATIONAL

    def test_navigational_wikilink(self):
        assert classify_query("see [[getting-started]]") == QueryType.NAVIGATIONAL

    def test_exploratory_what_question(self):
        assert classify_query("what is dependency injection") == QueryType.EXPLORATORY

    def test_exploratory_how_question(self):
        assert classify_query("how does caching work") == QueryType.EXPLORATORY

    def test_exploratory_why_question(self):
        assert classify_query("why use async") == QueryType.EXPLORATORY

    def test_exploratory_question_mark(self):
        assert classify_query("best practices for testing?") == QueryType.EXPLORATORY

    def test_exploratory_default(self):
        assert classify_query("performance optimization") == QueryType.EXPLORATORY

    def test_factual_priority_over_navigational(self):
        result = classify_query("getUserById in the utils section")
        assert result == QueryType.FACTUAL

    def test_factual_priority_over_exploratory(self):
        result = classify_query("what is get_user_by_id")
        assert result == QueryType.FACTUAL


class TestGetAdaptiveWeights:
    def test_factual_boosts_keyword(self):
        semantic, keyword, graph = get_adaptive_weights(
            QueryType.FACTUAL, 1.0, 0.8, 1.0
        )
        assert semantic == 1.0
        assert keyword == pytest.approx(1.2)
        assert graph == 1.0

    def test_navigational_boosts_graph(self):
        semantic, keyword, graph = get_adaptive_weights(
            QueryType.NAVIGATIONAL, 1.0, 0.8, 1.0
        )
        assert semantic == 1.0
        assert keyword == 0.8
        assert graph == pytest.approx(1.5)

    def test_exploratory_boosts_semantic(self):
        semantic, keyword, graph = get_adaptive_weights(
            QueryType.EXPLORATORY, 1.0, 0.8, 1.0
        )
        assert semantic == pytest.approx(1.3)
        assert keyword == 0.8
        assert graph == 1.0

    def test_preserves_custom_base_weights(self):
        semantic, keyword, graph = get_adaptive_weights(
            QueryType.FACTUAL, 2.0, 1.5, 0.5
        )
        assert semantic == 2.0
        assert keyword == pytest.approx(2.25)
        assert graph == 0.5
