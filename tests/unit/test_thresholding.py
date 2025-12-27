"""
Unit tests for score-based thresholding in context compression.

Tests cover:
- Basic filtering with mixed results above/below threshold
- Edge case: all results above threshold (no filtering)
- Edge case: all results below threshold (empty result)
- Edge case: empty input
- Edge case: threshold at boundary values (0.0, 1.0)
"""

import pytest

from src.compression.thresholding import filter_by_score
from src.models import ChunkResult


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def chunk_results() -> list[ChunkResult]:
    """
    Create a list of ChunkResult objects with varying scores.

    Provides results with scores: 0.9, 0.7, 0.5, 0.3, 0.1
    for testing various threshold scenarios.
    """
    return [
        ChunkResult(
            chunk_id="chunk_1",
            doc_id="doc_1",
            score=0.9,
            header_path="# High Score",
            file_path="/docs/high.md",
            content="High relevance content",
        ),
        ChunkResult(
            chunk_id="chunk_2",
            doc_id="doc_1",
            score=0.7,
            header_path="# Medium-High Score",
            file_path="/docs/medium_high.md",
            content="Medium-high relevance content",
        ),
        ChunkResult(
            chunk_id="chunk_3",
            doc_id="doc_2",
            score=0.5,
            header_path="# Medium Score",
            file_path="/docs/medium.md",
            content="Medium relevance content",
        ),
        ChunkResult(
            chunk_id="chunk_4",
            doc_id="doc_2",
            score=0.3,
            header_path="# Low Score",
            file_path="/docs/low.md",
            content="Low relevance content",
        ),
        ChunkResult(
            chunk_id="chunk_5",
            doc_id="doc_3",
            score=0.1,
            header_path="# Very Low Score",
            file_path="/docs/very_low.md",
            content="Very low relevance content",
        ),
    ]


# ============================================================================
# Basic Filtering Tests
# ============================================================================


class TestFilterByScoreBasic:
    """Tests for basic score filtering behavior."""

    def test_filter_by_score_mixed_results(
        self,
        chunk_results: list[ChunkResult],
    ) -> None:
        """
        Tests filtering when some results are above and some below threshold.

        With threshold 0.5, results with scores 0.9, 0.7, 0.5 should pass,
        while 0.3 and 0.1 should be filtered out.
        """
        filtered = filter_by_score(chunk_results, min_score=0.5)

        assert len(filtered) == 3
        assert all(r.score >= 0.5 for r in filtered)
        assert {r.chunk_id for r in filtered} == {"chunk_1", "chunk_2", "chunk_3"}

    def test_filter_preserves_order(
        self,
        chunk_results: list[ChunkResult],
    ) -> None:
        """
        Verifies that filtering preserves the original ordering of results.

        Results should maintain their relative order after filtering,
        which is important for downstream processing.
        """
        filtered = filter_by_score(chunk_results, min_score=0.4)

        scores = [r.score for r in filtered]
        assert scores == [0.9, 0.7, 0.5]

    def test_filter_uses_default_threshold(
        self,
        chunk_results: list[ChunkResult],
    ) -> None:
        """
        Tests that default threshold of 0.3 is applied when not specified.

        Default min_score=0.3 should include results at exactly 0.3
        and above, filtering out only the 0.1 result.
        """
        filtered = filter_by_score(chunk_results)

        assert len(filtered) == 4
        assert all(r.score >= 0.3 for r in filtered)


# ============================================================================
# Edge Case Tests
# ============================================================================


class TestFilterByScoreEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_all_results_above_threshold(
        self,
        chunk_results: list[ChunkResult],
    ) -> None:
        """
        Tests filtering when all results exceed the threshold.

        With a very low threshold (0.0), all results should pass through
        without any filtering.
        """
        filtered = filter_by_score(chunk_results, min_score=0.0)

        assert len(filtered) == len(chunk_results)
        assert filtered == chunk_results

    def test_all_results_below_threshold(
        self,
        chunk_results: list[ChunkResult],
    ) -> None:
        """
        Tests filtering when all results are below threshold.

        With a very high threshold (1.0), all results should be filtered out
        since no score equals exactly 1.0.
        """
        filtered = filter_by_score(chunk_results, min_score=1.0)

        assert len(filtered) == 0

    def test_empty_input(self) -> None:
        """
        Tests filtering with empty input list.

        Should return empty list without errors.
        """
        filtered = filter_by_score([], min_score=0.5)

        assert filtered == []

    def test_threshold_at_exact_score_boundary(
        self,
        chunk_results: list[ChunkResult],
    ) -> None:
        """
        Tests filtering when threshold exactly matches a result's score.

        Result with score exactly at threshold should be included
        (using >= comparison).
        """
        # Threshold exactly matches chunk_4's score of 0.3
        filtered = filter_by_score(chunk_results, min_score=0.3)

        assert len(filtered) == 4
        assert any(r.score == 0.3 for r in filtered)
        chunk_ids = {r.chunk_id for r in filtered}
        assert "chunk_4" in chunk_ids

    def test_single_result_above_threshold(self) -> None:
        """
        Tests filtering a single result that meets the threshold.

        Should return the single result in a list.
        """
        single_result = [
            ChunkResult(
                chunk_id="single",
                doc_id="doc",
                score=0.8,
                header_path="# Single",
                file_path="/docs/single.md",
                content="Single result",
            )
        ]

        filtered = filter_by_score(single_result, min_score=0.5)

        assert len(filtered) == 1
        assert filtered[0].chunk_id == "single"

    def test_single_result_below_threshold(self) -> None:
        """
        Tests filtering a single result that does not meet the threshold.

        Should return empty list.
        """
        single_result = [
            ChunkResult(
                chunk_id="single",
                doc_id="doc",
                score=0.2,
                header_path="# Single",
                file_path="/docs/single.md",
                content="Single result",
            )
        ]

        filtered = filter_by_score(single_result, min_score=0.5)

        assert len(filtered) == 0


# ============================================================================
# Boundary Value Tests
# ============================================================================


class TestFilterByScoreBoundaryValues:
    """Tests for extreme and boundary threshold values."""

    def test_threshold_zero_includes_all(self) -> None:
        """
        Tests that threshold of 0.0 includes all results.

        A score of 0.0 or greater means everything passes.
        """
        results = [
            ChunkResult(
                chunk_id=f"chunk_{i}",
                doc_id="doc",
                score=i * 0.1,
                header_path=f"# Score {i * 0.1}",
                file_path=f"/docs/score_{i}.md",
                content=f"Score {i * 0.1} content",
            )
            for i in range(11)  # scores 0.0, 0.1, ..., 1.0
        ]

        filtered = filter_by_score(results, min_score=0.0)

        assert len(filtered) == 11

    def test_threshold_one_very_restrictive(self) -> None:
        """
        Tests that threshold of 1.0 only includes perfect scores.

        Only results with exactly score=1.0 should pass.
        """
        results = [
            ChunkResult(
                chunk_id="perfect",
                doc_id="doc",
                score=1.0,
                header_path="# Perfect",
                file_path="/docs/perfect.md",
                content="Perfect score",
            ),
            ChunkResult(
                chunk_id="almost",
                doc_id="doc",
                score=0.99,
                header_path="# Almost",
                file_path="/docs/almost.md",
                content="Almost perfect",
            ),
        ]

        filtered = filter_by_score(results, min_score=1.0)

        assert len(filtered) == 1
        assert filtered[0].chunk_id == "perfect"

    def test_negative_scores_handled(self) -> None:
        """
        Tests handling of results with negative scores.

        Negative scores (if they somehow occur) should be filtered
        by any positive threshold.
        """
        results = [
            ChunkResult(
                chunk_id="negative",
                doc_id="doc",
                score=-0.5,
                header_path="# Negative",
                file_path="/docs/negative.md",
                content="Negative score",
            ),
            ChunkResult(
                chunk_id="positive",
                doc_id="doc",
                score=0.5,
                header_path="# Positive",
                file_path="/docs/positive.md",
                content="Positive score",
            ),
        ]

        filtered = filter_by_score(results, min_score=0.0)

        assert len(filtered) == 1
        assert filtered[0].chunk_id == "positive"
