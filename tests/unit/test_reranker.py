"""
Unit tests for ReRanker class with cross-encoder.

Tests cover:
- Empty candidate handling
- Re-ranking scores relevant docs higher
- Preserves count constraint (top_n)
- Graceful handling when chunk content not found
- Lazy model loading behavior

Test strategies:
- Use a FakeCrossEncoder for deterministic, fast tests without model downloads
- Test both happy path and edge cases
- Verify lazy loading by checking model state
"""

from dataclasses import dataclass

import pytest

from src.search.reranker import ReRanker


@dataclass
class FakeCrossEncoder:
    def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        return [len(content) * 0.01 for _, content in pairs]


@pytest.fixture
def fake_cross_encoder():
    """
    Create a FakeCrossEncoder that returns predictable scores.

    Scores are based on content length to make tests deterministic.
    Longer content = higher score (simulating relevance).
    """
    return FakeCrossEncoder()


@pytest.fixture
def reranker_with_fake(fake_cross_encoder: FakeCrossEncoder):
    """
    Create a ReRanker with injected fake model.

    Directly injects the fake model to avoid model download.
    Uses internal state injection since CrossEncoder is loaded lazily.
    """
    reranker = ReRanker(model_name="test-model")
    reranker._model = fake_cross_encoder
    return reranker


@pytest.fixture
def content_provider():
    """
    Provide test content of varying lengths.
    """
    contents = {
        "chunk_1": "Short content",
        "chunk_2": "Medium length content for testing purposes",
        "chunk_3": "This is a much longer content that should score higher when reranking",
        "chunk_4": "Tiny",
    }
    return lambda chunk_id: contents.get(chunk_id)


@pytest.fixture
def candidates():
    """
    Sample candidates for re-ranking.
    """
    return [
        ("chunk_1", 0.9),
        ("chunk_2", 0.7),
        ("chunk_3", 0.5),
        ("chunk_4", 0.3),
    ]


class TestReRankEmptyCandidates:

    def test_rerank_empty_candidates_returns_empty(self, reranker_with_fake, content_provider):
        """
        Empty candidate list returns empty list.
        """
        result = reranker_with_fake.rerank(
            query="test query",
            candidates=[],
            content_provider=content_provider,
            top_n=5,
        )
        assert result == []

    def test_rerank_empty_does_not_load_model(self):
        """
        Re-ranking empty list does not trigger model loading.
        """
        reranker = ReRanker(model_name="test-model")
        assert reranker._model is None

        result = reranker.rerank(
            query="test query",
            candidates=[],
            content_provider=lambda x: None,
            top_n=5,
        )
        assert result == []
        assert reranker._model is None


class TestReRankScoring:

    def test_rerank_scores_relevant_higher(
        self, fake_cross_encoder: FakeCrossEncoder, content_provider, candidates
    ):
        """
        Cross-encoder scores more relevant docs higher.
        """
        reranker = ReRanker(model_name="test-model")
        reranker._model = fake_cross_encoder

        result = reranker.rerank(
            query="test query",
            candidates=candidates,
            content_provider=content_provider,
            top_n=10,
        )

        assert result[0][0] == "chunk_3"
        assert result[-1][0] == "chunk_4"

    def test_rerank_returns_float_scores(
        self, reranker_with_fake, content_provider, candidates
    ):
        """
        Reranked results have float scores.
        """
        result = reranker_with_fake.rerank(
            query="test query",
            candidates=candidates,
            content_provider=content_provider,
            top_n=10,
        )

        for chunk_id, score in result:
            assert isinstance(chunk_id, str)
            assert isinstance(score, float)


class TestReRankCountPreservation:

    def test_rerank_preserves_count_returns_top_n(
        self, reranker_with_fake, content_provider, candidates
    ):
        """
        Returns at most top_n results.
        """
        result = reranker_with_fake.rerank(
            query="test query",
            candidates=candidates,
            content_provider=content_provider,
            top_n=2,
        )
        assert len(result) == 2

    def test_rerank_fewer_than_top_n(self, reranker_with_fake, content_provider):
        """
        Returns all candidates when fewer than top_n.
        """
        small_candidates = [("chunk_1", 0.9), ("chunk_2", 0.7)]

        result = reranker_with_fake.rerank(
            query="test query",
            candidates=small_candidates,
            content_provider=content_provider,
            top_n=5,
        )
        assert len(result) == 2


class TestReRankMissingContent:

    def test_rerank_handles_missing_content(self, fake_cross_encoder: FakeCrossEncoder):
        """
        Gracefully handles chunks where content not found.
        """
        def partial_provider(chunk_id: str) -> str | None:
            return "Found content" if chunk_id == "chunk_1" else None

        candidates = [
            ("chunk_1", 0.9),
            ("chunk_2", 0.7),
        ]

        reranker = ReRanker(model_name="test-model")
        reranker._model = fake_cross_encoder

        result = reranker.rerank(
            query="test query",
            candidates=candidates,
            content_provider=partial_provider,
            top_n=10,
        )

        result_ids = {cid for cid, _ in result}
        assert "chunk_1" in result_ids
        assert "chunk_2" in result_ids

    def test_rerank_all_missing_content_returns_original(
        self, fake_cross_encoder: FakeCrossEncoder, candidates
    ):
        """
        When all content missing, returns original candidates truncated.
        """
        reranker = ReRanker(model_name="test-model")
        reranker._model = fake_cross_encoder

        result = reranker.rerank(
            query="test query",
            candidates=candidates,
            content_provider=lambda x: None,
            top_n=2,
        )
        assert len(result) == 2


class TestLazyModelLoading:

    def test_lazy_model_loading_not_loaded_on_init(self):
        """
        Model not loaded until first rerank call.
        """
        reranker = ReRanker(model_name="test-model")
        assert reranker._model is None
        assert reranker._model_name == "test-model"

    def test_model_reused_across_calls(
        self, fake_cross_encoder: FakeCrossEncoder, content_provider
    ):
        """
        Once injected, model is reused across multiple rerank calls.
        """
        reranker = ReRanker(model_name="test-model")
        reranker._model = fake_cross_encoder
        candidates = [("chunk_1", 0.9)]

        reranker.rerank("query1", candidates, content_provider, 5)
        reranker.rerank("query2", candidates, content_provider, 5)
        reranker.rerank("query3", candidates, content_provider, 5)

        assert reranker._model is fake_cross_encoder


class TestReRankEdgeCases:

    def test_rerank_single_candidate(self, reranker_with_fake, content_provider):
        """
        Single candidate is handled correctly.
        """
        single_candidate = [("chunk_1", 0.9)]

        result = reranker_with_fake.rerank(
            query="test query",
            candidates=single_candidate,
            content_provider=content_provider,
            top_n=5,
        )

        assert len(result) == 1
        assert result[0][0] == "chunk_1"

    def test_rerank_top_n_zero(self, reranker_with_fake, content_provider, candidates):
        """
        top_n=0 returns empty list.
        """
        result = reranker_with_fake.rerank(
            query="test query",
            candidates=candidates,
            content_provider=content_provider,
            top_n=0,
        )
        assert result == []

    def test_rerank_preserves_chunk_ids(
        self, reranker_with_fake, content_provider, candidates
    ):
        """
        Chunk IDs are preserved through reranking.
        """
        result = reranker_with_fake.rerank(
            query="test query",
            candidates=candidates,
            content_provider=content_provider,
            top_n=10,
        )

        input_ids = {cid for cid, _ in candidates}
        output_ids = {cid for cid, _ in result}
        assert output_ids == input_ids
