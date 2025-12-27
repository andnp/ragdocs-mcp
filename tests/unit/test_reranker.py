"""
Unit tests for ReRanker class with cross-encoder.

Tests cover:
- Empty candidate handling
- Re-ranking scores relevant docs higher
- Preserves count constraint (top_n)
- Graceful handling when chunk content not found
- Lazy model loading behavior

Test strategies:
- Mock the CrossEncoder to avoid downloading real models in unit tests
- Test both happy path and edge cases
- Verify lazy loading by checking model state
"""

from unittest.mock import MagicMock, patch

import pytest

from src.search.reranker import ReRanker


# Patch target for CrossEncoder (imported inside _ensure_model_loaded)
CROSS_ENCODER_PATCH = "sentence_transformers.CrossEncoder"


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def mock_cross_encoder():
    """
    Create a mock CrossEncoder that returns predictable scores.

    Scores are based on content length to make tests deterministic.
    Longer content = higher score (simulating relevance).
    """
    mock = MagicMock()
    mock.predict = MagicMock(
        side_effect=lambda pairs: [len(content) * 0.01 for _, content in pairs]
    )
    return mock


@pytest.fixture
def reranker_with_mock(mock_cross_encoder):
    """
    Create a ReRanker with injected mock model.

    Directly injects the mock model to avoid model download.
    Uses internal state injection since CrossEncoder is loaded lazily.
    """
    reranker = ReRanker(model_name="test-model")
    # Directly inject the mock model to bypass lazy loading
    reranker._model = mock_cross_encoder
    return reranker


@pytest.fixture
def content_provider():
    """
    Create a content provider function that returns mock content.

    Returns content of varying lengths to test scoring behavior.
    """
    contents = {
        "chunk_1": "Short content",  # 13 chars
        "chunk_2": "Medium length content for testing purposes",  # 43 chars
        "chunk_3": "This is a much longer content that should score higher when reranking",  # 70 chars
        "chunk_4": "Tiny",  # 4 chars
    }
    return lambda chunk_id: contents.get(chunk_id)


@pytest.fixture
def candidates():
    """
    Create sample candidates for re-ranking.

    Returns list of (chunk_id, original_score) tuples.
    Original scores are ordered differently from expected reranked order.
    """
    return [
        ("chunk_1", 0.9),  # Short content, high original score
        ("chunk_2", 0.7),  # Medium content
        ("chunk_3", 0.5),  # Long content, low original score
        ("chunk_4", 0.3),  # Tiny content
    ]


# ============================================================================
# Empty Candidates Tests
# ============================================================================


class TestReRankEmptyCandidates:
    """Tests for handling empty candidate lists."""

    def test_rerank_empty_candidates_returns_empty(self, reranker_with_mock, content_provider):
        """
        Empty candidate list returns empty list.

        Verifies no errors when reranking empty input.
        """
        result = reranker_with_mock.rerank(
            query="test query",
            candidates=[],
            content_provider=content_provider,
            top_n=5,
        )

        assert result == []

    def test_rerank_empty_does_not_load_model(self):
        """
        Re-ranking empty list does not trigger model loading.

        Verifies lazy loading optimization - model should only load
        when actually needed.
        """
        reranker = ReRanker(model_name="test-model")

        # Model should not be loaded yet
        assert reranker._model is None

        def empty_provider(chunk_id: str) -> str | None:
            return None

        result = reranker.rerank(
            query="test query",
            candidates=[],
            content_provider=empty_provider,
            top_n=5,
        )

        # Model should still not be loaded for empty input
        assert result == []
        assert reranker._model is None


# ============================================================================
# Scoring Tests
# ============================================================================


class TestReRankScoring:
    """Tests for re-ranking scoring behavior."""

    def test_rerank_scores_relevant_higher(
        self,
        mock_cross_encoder,
        content_provider,
        candidates,
    ):
        """
        Cross-encoder scores more relevant docs higher.

        With mock scoring based on content length, longer content
        should rank higher after re-ranking.
        """
        with patch(CROSS_ENCODER_PATCH, return_value=mock_cross_encoder):
            reranker = ReRanker(model_name="test-model")

            result = reranker.rerank(
                query="test query",
                candidates=candidates,
                content_provider=content_provider,
                top_n=10,
            )

        # Verify results are sorted by cross-encoder scores (content length)
        # chunk_3 has longest content, should be first
        assert result[0][0] == "chunk_3"
        # chunk_4 has shortest content, should be last
        assert result[-1][0] == "chunk_4"

    def test_rerank_returns_float_scores(
        self,
        reranker_with_mock,
        content_provider,
        candidates,
    ):
        """
        Reranked results have float scores.

        Verifies score type after re-ranking.
        """
        result = reranker_with_mock.rerank(
            query="test query",
            candidates=candidates,
            content_provider=content_provider,
            top_n=10,
        )

        for chunk_id, score in result:
            assert isinstance(chunk_id, str)
            assert isinstance(score, float)


# ============================================================================
# Count Preservation Tests
# ============================================================================


class TestReRankCountPreservation:
    """Tests for top_n count handling."""

    def test_rerank_preserves_count_returns_top_n(
        self,
        reranker_with_mock,
        content_provider,
        candidates,
    ):
        """
        Returns at most top_n results.

        Verifies count constraint is respected.
        """
        result = reranker_with_mock.rerank(
            query="test query",
            candidates=candidates,
            content_provider=content_provider,
            top_n=2,
        )

        assert len(result) == 2

    def test_rerank_fewer_than_top_n(
        self,
        reranker_with_mock,
        content_provider,
    ):
        """
        Returns all candidates when fewer than top_n.

        If only 2 candidates exist and top_n=5, should return 2.
        """
        small_candidates = [("chunk_1", 0.9), ("chunk_2", 0.7)]

        result = reranker_with_mock.rerank(
            query="test query",
            candidates=small_candidates,
            content_provider=content_provider,
            top_n=5,
        )

        assert len(result) == 2


# ============================================================================
# Missing Content Tests
# ============================================================================


class TestReRankMissingContent:
    """Tests for handling missing chunk content."""

    def test_rerank_handles_missing_content(
        self,
        mock_cross_encoder,
    ):
        """
        Gracefully handles chunks where content not found.

        Chunks without content should still be included but with
        penalized scores.
        """
        def partial_provider(chunk_id: str) -> str | None:
            return "Found content" if chunk_id == "chunk_1" else None

        candidates = [
            ("chunk_1", 0.9),  # Content found
            ("chunk_2", 0.7),  # Content NOT found
        ]

        with patch(CROSS_ENCODER_PATCH, return_value=mock_cross_encoder):
            reranker = ReRanker(model_name="test-model")

            result = reranker.rerank(
                query="test query",
                candidates=candidates,
                content_provider=partial_provider,
                top_n=10,
            )

        # Both chunks should be in results
        result_ids = {cid for cid, _ in result}
        assert "chunk_1" in result_ids
        assert "chunk_2" in result_ids

    def test_rerank_all_missing_content_returns_original(
        self,
        mock_cross_encoder,
        candidates,
    ):
        """
        When all content missing, returns original candidates truncated.

        Fallback behavior when content provider returns None for all.
        """
        def empty_provider(chunk_id: str) -> str | None:
            return None

        with patch(CROSS_ENCODER_PATCH, return_value=mock_cross_encoder):
            reranker = ReRanker(model_name="test-model")

            result = reranker.rerank(
                query="test query",
                candidates=candidates,
                content_provider=empty_provider,
                top_n=2,
            )

        # Should return top_n candidates with penalized scores
        assert len(result) == 2


# ============================================================================
# Lazy Model Loading Tests
# ============================================================================


class TestLazyModelLoading:
    """Tests for lazy model loading behavior."""

    def test_lazy_model_loading_not_loaded_on_init(self):
        """
        Model not loaded until first rerank call.

        Verifies lazy loading - constructor should not trigger download.
        """
        reranker = ReRanker(model_name="test-model")

        assert reranker._model is None
        assert reranker._model_name == "test-model"

    def test_lazy_model_loading_loads_on_first_call(
        self,
        mock_cross_encoder,
        content_provider,
    ):
        """
        Model loaded on first non-empty rerank call.

        Verifies model is loaded when actually needed.
        """
        with patch(CROSS_ENCODER_PATCH, return_value=mock_cross_encoder) as mock_cls:
            reranker = ReRanker(model_name="test-model")

            # Model not loaded yet
            assert reranker._model is None
            mock_cls.assert_not_called()

            # First call with non-empty candidates should load model
            candidates = [("chunk_1", 0.9)]
            reranker.rerank(
                query="test query",
                candidates=candidates,
                content_provider=content_provider,
                top_n=5,
            )

            # Model should now be loaded
            mock_cls.assert_called_once_with("test-model")
            assert reranker._model is not None

    def test_lazy_model_loading_reuses_model(
        self,
        mock_cross_encoder,
        content_provider,
    ):
        """
        Model is reused across multiple rerank calls.

        Verifies model loaded only once even with multiple calls.
        """
        with patch(CROSS_ENCODER_PATCH, return_value=mock_cross_encoder) as mock_cls:
            reranker = ReRanker(model_name="test-model")
            candidates = [("chunk_1", 0.9)]

            # Multiple calls
            reranker.rerank("query1", candidates, content_provider, 5)
            reranker.rerank("query2", candidates, content_provider, 5)
            reranker.rerank("query3", candidates, content_provider, 5)

            # Model should only be created once
            mock_cls.assert_called_once()


# ============================================================================
# Edge Case Tests
# ============================================================================


class TestReRankEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_rerank_single_candidate(
        self,
        reranker_with_mock,
        content_provider,
    ):
        """
        Single candidate is handled correctly.

        Should return the single candidate reranked.
        """
        single_candidate = [("chunk_1", 0.9)]

        result = reranker_with_mock.rerank(
            query="test query",
            candidates=single_candidate,
            content_provider=content_provider,
            top_n=5,
        )

        assert len(result) == 1
        assert result[0][0] == "chunk_1"

    def test_rerank_top_n_zero(
        self,
        reranker_with_mock,
        content_provider,
        candidates,
    ):
        """
        top_n=0 returns empty list.

        Edge case where caller requests zero results.
        """
        result = reranker_with_mock.rerank(
            query="test query",
            candidates=candidates,
            content_provider=content_provider,
            top_n=0,
        )

        assert result == []

    def test_rerank_preserves_chunk_ids(
        self,
        reranker_with_mock,
        content_provider,
        candidates,
    ):
        """
        Chunk IDs are preserved through reranking.

        Original chunk IDs should appear in output.
        """
        result = reranker_with_mock.rerank(
            query="test query",
            candidates=candidates,
            content_provider=content_provider,
            top_n=10,
        )

        input_ids = {cid for cid, _ in candidates}
        output_ids = {cid for cid, _ in result}

        assert output_ids == input_ids
