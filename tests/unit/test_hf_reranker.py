"""Unit tests for HuggingFaceReranker port conformance and basic functionality.

Tests that the adapter:
1. Conforms to the Reranker protocol
2. Scores relevant documents higher than irrelevant ones
"""

import pytest

from searchkernel.adapters.rerank import HuggingFaceReranker
from searchkernel.ports import Reranker


class TestHuggingFaceRerankerConformance:
    """Test port conformance."""

    def test_implements_reranker_protocol(self) -> None:
        """HuggingFaceReranker implements the Reranker protocol."""
        reranker = HuggingFaceReranker()
        assert isinstance(reranker, Reranker)

    def test_has_model_name_attribute(self) -> None:
        """Has model_name attribute."""
        reranker = HuggingFaceReranker()
        assert hasattr(reranker, "model_name")
        assert isinstance(reranker.model_name, str)
        assert "Qwen3-Reranker" in reranker.model_name

    def test_has_rerank_method(self) -> None:
        """Has rerank method with correct signature."""
        reranker = HuggingFaceReranker()
        assert hasattr(reranker, "rerank")
        assert callable(reranker.rerank)


class TestHuggingFaceRerankerScoring:
    """Test scoring behavior."""

    @pytest.fixture
    def reranker(self) -> HuggingFaceReranker:
        """Create a reranker instance."""
        return HuggingFaceReranker()

    def test_rerank_returns_list_of_floats(self, reranker: HuggingFaceReranker) -> None:
        """rerank returns a list of floats."""
        query = "What is Python?"
        documents = ["Python is a programming language.", "A snake is a reptile."]
        scores = reranker.rerank(query, documents)

        assert isinstance(scores, list)
        assert len(scores) == len(documents)
        assert all(isinstance(s, float) for s in scores)

    def test_rerank_scores_in_valid_range(self, reranker: HuggingFaceReranker) -> None:
        """Scores are in the [0, 1] range."""
        query = "What is Python?"
        documents = ["Python is a programming language.", "A snake is a reptile."]
        scores = reranker.rerank(query, documents)

        assert all(0 <= s <= 1 for s in scores), f"Scores out of range: {scores}"

    def test_rerank_relevant_higher_than_irrelevant(
        self, reranker: HuggingFaceReranker
    ) -> None:
        """Relevant documents score higher than irrelevant ones.

        A clearly relevant document (about Python) should score higher
        than an irrelevant one (about snakes) for a Python query.
        """
        query = "What is Python programming language?"
        relevant = "Python is a high-level, interpreted programming language created by Guido van Rossum."
        irrelevant = "A snake is a legless reptile found in many parts of the world."

        scores = reranker.rerank(query, [relevant, irrelevant])
        relevant_score, irrelevant_score = scores

        assert (
            relevant_score > irrelevant_score
        ), f"Expected relevant ({relevant_score}) > irrelevant ({irrelevant_score})"

    def test_rerank_maintains_order(self, reranker: HuggingFaceReranker) -> None:
        """Output scores are in the same order as input documents."""
        query = "machine learning"
        documents = [
            "Machine learning is a subset of AI.",
            "Cats are adorable pets.",
            "Deep learning uses neural networks.",
        ]
        scores = reranker.rerank(query, documents)

        assert len(scores) == 3
        # Documents 0 and 2 are relevant, document 1 is not
        assert scores[0] > scores[1]  # ML > Cats
        assert scores[2] > scores[1]  # Deep learning > Cats
