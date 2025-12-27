"""
Unit tests for search filtering functions.
"""

from src.search.filters import filter_by_confidence, limit_per_document


class TestFilterByConfidence:
    """Tests for filter_by_confidence function."""

    def test_empty_list_returns_empty(self):
        """Empty input returns empty output."""
        result = filter_by_confidence([], threshold=0.5)
        assert result == []

    def test_threshold_zero_returns_all(self):
        """Threshold 0.0 disables filtering, returns all results."""
        results = [
            ("chunk_a", 0.9),
            ("chunk_b", 0.5),
            ("chunk_c", 0.1),
        ]
        filtered = filter_by_confidence(results, threshold=0.0)
        assert filtered == results

    def test_threshold_negative_returns_all(self):
        """Negative threshold acts same as zero (disabled)."""
        results = [
            ("chunk_a", 0.9),
            ("chunk_b", 0.1),
        ]
        filtered = filter_by_confidence(results, threshold=-0.5)
        assert filtered == results

    def test_filters_below_threshold(self):
        """Results below threshold are removed."""
        results = [
            ("chunk_a", 0.9),
            ("chunk_b", 0.5),
            ("chunk_c", 0.3),
            ("chunk_d", 0.1),
        ]
        filtered = filter_by_confidence(results, threshold=0.4)
        assert filtered == [("chunk_a", 0.9), ("chunk_b", 0.5)]

    def test_all_above_threshold(self):
        """When all results are above threshold, all are kept."""
        results = [
            ("chunk_a", 0.9),
            ("chunk_b", 0.8),
            ("chunk_c", 0.7),
        ]
        filtered = filter_by_confidence(results, threshold=0.5)
        assert filtered == results

    def test_all_below_threshold(self):
        """When all results are below threshold, empty list returned."""
        results = [
            ("chunk_a", 0.3),
            ("chunk_b", 0.2),
            ("chunk_c", 0.1),
        ]
        filtered = filter_by_confidence(results, threshold=0.5)
        assert filtered == []

    def test_exactly_at_threshold_included(self):
        """Results exactly at threshold are included (>= comparison)."""
        results = [
            ("chunk_a", 0.5),
            ("chunk_b", 0.49),
        ]
        filtered = filter_by_confidence(results, threshold=0.5)
        assert filtered == [("chunk_a", 0.5)]

    def test_default_threshold_is_zero(self):
        """Default threshold is 0.0 (returns all)."""
        results = [("chunk_a", 0.1)]
        filtered = filter_by_confidence(results)
        assert filtered == results


class TestLimitPerDocument:
    """Tests for limit_per_document function."""

    def test_empty_list_returns_empty(self):
        """Empty input returns empty output."""
        result = limit_per_document([], max_per_doc=2)
        assert result == []

    def test_max_per_doc_zero_returns_all(self):
        """max_per_doc=0 disables limiting, returns all results."""
        results = [
            ("doc1_chunk_0", 0.9),
            ("doc1_chunk_1", 0.8),
            ("doc1_chunk_2", 0.7),
        ]
        limited = limit_per_document(results, max_per_doc=0)
        assert limited == results

    def test_max_per_doc_negative_returns_all(self):
        """Negative max_per_doc acts same as zero (disabled)."""
        results = [
            ("doc1_chunk_0", 0.9),
            ("doc1_chunk_1", 0.8),
        ]
        limited = limit_per_document(results, max_per_doc=-1)
        assert limited == results

    def test_limits_correctly_per_document(self):
        """Limits results per document correctly."""
        results = [
            ("doc1_chunk_0", 0.95),
            ("doc1_chunk_1", 0.90),
            ("doc1_chunk_2", 0.85),  # Should be excluded
            ("doc2_chunk_0", 0.80),
            ("doc2_chunk_1", 0.75),
            ("doc2_chunk_2", 0.70),  # Should be excluded
        ]
        limited = limit_per_document(results, max_per_doc=2)

        assert limited == [
            ("doc1_chunk_0", 0.95),
            ("doc1_chunk_1", 0.90),
            ("doc2_chunk_0", 0.80),
            ("doc2_chunk_1", 0.75),
        ]

    def test_max_per_doc_one(self):
        """max_per_doc=1 keeps only first chunk per document."""
        results = [
            ("doc1_chunk_0", 0.9),
            ("doc1_chunk_1", 0.8),
            ("doc2_chunk_0", 0.7),
            ("doc3_chunk_0", 0.6),
            ("doc3_chunk_1", 0.5),
        ]
        limited = limit_per_document(results, max_per_doc=1)

        assert limited == [
            ("doc1_chunk_0", 0.9),
            ("doc2_chunk_0", 0.7),
            ("doc3_chunk_0", 0.6),
        ]

    def test_fewer_chunks_than_limit(self):
        """Documents with fewer chunks than limit keep all."""
        results = [
            ("doc1_chunk_0", 0.9),
            ("doc2_chunk_0", 0.8),
            ("doc2_chunk_1", 0.7),
        ]
        limited = limit_per_document(results, max_per_doc=5)
        assert limited == results

    def test_handles_chunk_id_without_separator(self):
        """Chunk IDs without _chunk_ separator use full ID as doc_id."""
        results = [
            ("simple-doc-1", 0.9),
            ("simple-doc-2", 0.8),
            ("simple-doc-3", 0.7),
        ]
        limited = limit_per_document(results, max_per_doc=2)
        # Each is treated as a separate document
        assert limited == results

    def test_handles_mixed_chunk_id_formats(self):
        """Handles mix of chunked and non-chunked IDs."""
        results = [
            ("doc1_chunk_0", 0.9),
            ("doc1_chunk_1", 0.8),
            ("simple-doc", 0.7),  # No _chunk_ separator
            ("doc2_chunk_0", 0.6),
        ]
        limited = limit_per_document(results, max_per_doc=1)

        assert limited == [
            ("doc1_chunk_0", 0.9),
            ("simple-doc", 0.7),
            ("doc2_chunk_0", 0.6),
        ]

    def test_preserves_order(self):
        """Results maintain their original order after limiting."""
        results = [
            ("doc2_chunk_0", 0.9),
            ("doc1_chunk_0", 0.85),
            ("doc2_chunk_1", 0.8),
            ("doc1_chunk_1", 0.75),
        ]
        limited = limit_per_document(results, max_per_doc=1)

        # Order preserved: doc2 first, then doc1
        assert limited == [
            ("doc2_chunk_0", 0.9),
            ("doc1_chunk_0", 0.85),
        ]

    def test_default_max_per_doc_is_zero(self):
        """Default max_per_doc is 0 (returns all)."""
        results = [
            ("doc1_chunk_0", 0.9),
            ("doc1_chunk_1", 0.8),
        ]
        limited = limit_per_document(results)
        assert limited == results
