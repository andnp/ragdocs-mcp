"""Unit tests for semantic tier derivation from indexing progress."""

import pytest

from searchkernel.indexing.runtime_readiness import semantic_tier_from_progress


class TestSemanticTierFromProgress:
    """Test semantic tier state derivation from indexing progress."""

    def test_no_work_pending_when_total_zero(self) -> None:
        """When total_count is 0, no work is pending."""
        tier = semantic_tier_from_progress(indexed_count=0, total_count=0)
        assert tier == "available"

    def test_complete_when_all_indexed(self) -> None:
        """When indexed_count equals total_count, all work is complete."""
        tier = semantic_tier_from_progress(indexed_count=10, total_count=10)
        assert tier == "complete"

    def test_backfilling_when_partial(self) -> None:
        """When 0 < indexed_count < total_count, work is still backfilling."""
        tier = semantic_tier_from_progress(indexed_count=5, total_count=10)
        assert tier == "backfilling"

    def test_complete_when_indexed_exceeds_total(self) -> None:
        """When indexed_count >= total_count, all work is complete (edge case)."""
        tier = semantic_tier_from_progress(indexed_count=15, total_count=10)
        assert tier == "complete"

    def test_backfilling_at_boundary_start(self) -> None:
        """Test backfilling at the start of indexing (indexed_count=1)."""
        tier = semantic_tier_from_progress(indexed_count=1, total_count=10)
        assert tier == "backfilling"

    def test_backfilling_at_boundary_near_end(self) -> None:
        """Test backfilling near completion (indexed_count=total_count-1)."""
        tier = semantic_tier_from_progress(indexed_count=9, total_count=10)
        assert tier == "backfilling"

    @pytest.mark.parametrize(
        "indexed,total,expected",
        [
            (0, 0, "available"),
            (0, 1, "backfilling"),
            (1, 1, "complete"),
            (1, 2, "backfilling"),
            (2, 2, "complete"),
            (100, 100, "complete"),
            (99, 100, "backfilling"),
            (1, 100, "backfilling"),
        ],
    )
    def test_all_combinations(
        self, indexed: int, total: int, expected: str
    ) -> None:
        """Test a variety of progress combinations."""
        tier = semantic_tier_from_progress(indexed, total)
        assert tier == expected
