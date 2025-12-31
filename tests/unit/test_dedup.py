"""
Unit tests for deduplication functions.
"""

import numpy as np
import pytest

from src.search.dedup import (
    cosine_similarity,
    deduplicate_by_content_hash,
    deduplicate_by_ngram,
    deduplicate_by_similarity,
    get_ngrams,
    jaccard_similarity,
)


class TestGetNgrams:
    """Tests for get_ngrams function."""

    def test_basic_ngrams(self):
        """Extract 3-grams from simple text."""
        ngrams = get_ngrams("hello", n=3)
        assert ngrams == {"hel", "ell", "llo"}

    def test_empty_string_returns_empty_set(self):
        """Empty string returns empty set."""
        ngrams = get_ngrams("", n=3)
        assert ngrams == set()

    def test_short_string_returns_whole_string(self):
        """String shorter than n returns whole string as single ngram."""
        ngrams = get_ngrams("ab", n=3)
        assert ngrams == {"ab"}

    def test_string_equal_to_n_returns_single_ngram(self):
        """String equal to n returns single ngram."""
        ngrams = get_ngrams("abc", n=3)
        assert ngrams == {"abc"}

    def test_converts_to_lowercase(self):
        """Text is converted to lowercase."""
        ngrams = get_ngrams("HELLO", n=3)
        assert ngrams == {"hel", "ell", "llo"}

    def test_strips_whitespace(self):
        """Whitespace is stripped from edges."""
        ngrams = get_ngrams("  hi  ", n=2)
        assert ngrams == {"hi"}

    def test_custom_n_value(self):
        """Works with different n values."""
        ngrams = get_ngrams("hello", n=2)
        assert ngrams == {"he", "el", "ll", "lo"}


class TestJaccardSimilarity:
    """Tests for jaccard_similarity function."""

    def test_identical_texts_returns_one(self):
        """Identical texts have similarity 1.0."""
        sim = jaccard_similarity("hello world", "hello world")
        assert sim == pytest.approx(1.0)

    def test_completely_different_returns_zero(self):
        """Completely different texts have similarity 0.0."""
        sim = jaccard_similarity("aaa", "bbb", n=3)
        assert sim == pytest.approx(0.0)

    def test_partial_overlap(self):
        """Texts with partial overlap have intermediate similarity."""
        sim = jaccard_similarity("hello", "hella", n=3)
        # "hel", "ell" overlap; "llo" vs "lla" differ
        # hello: {hel, ell, llo}, hella: {hel, ell, lla}
        # intersection = 2, union = 4
        assert sim == pytest.approx(0.5)

    def test_empty_text_returns_zero(self):
        """Empty text returns similarity 0.0."""
        sim = jaccard_similarity("", "hello")
        assert sim == pytest.approx(0.0)

    def test_both_empty_returns_zero(self):
        """Both empty texts return 0.0."""
        sim = jaccard_similarity("", "")
        assert sim == pytest.approx(0.0)


class TestDeduplicateByNgram:
    """Tests for deduplicate_by_ngram function."""

    def test_empty_list_returns_empty(self):
        """Empty input returns empty list and 0 removed."""
        def get_content(chunk_id: str):
            return None

        result, removed = deduplicate_by_ngram([], get_content)

        assert result == []
        assert removed == 0

    def test_single_item_returns_unchanged(self):
        """Single item returns unchanged, 0 removed."""
        def get_content(chunk_id: str):
            return "some content"

        results = [("chunk_a", 0.9)]
        deduped, removed = deduplicate_by_ngram(results, get_content)

        assert deduped == [("chunk_a", 0.9)]
        assert removed == 0

    def test_removes_similar_content(self):
        """Removes chunks with high n-gram overlap."""
        contents = {
            "chunk_a": "The quick brown fox jumps over the lazy dog",
            "chunk_b": "The quick brown fox jumps over the lazy cat",  # Very similar
            "chunk_c": "A completely different text about programming",
        }

        def get_content(chunk_id: str):
            return contents.get(chunk_id)

        results = [
            ("chunk_a", 0.9),
            ("chunk_b", 0.8),
            ("chunk_c", 0.7),
        ]

        deduped, removed = deduplicate_by_ngram(results, get_content, threshold=0.7)

        # chunk_b should be removed as too similar to chunk_a
        assert len(deduped) == 2
        assert ("chunk_a", 0.9) in deduped
        assert ("chunk_c", 0.7) in deduped
        assert removed == 1

    def test_keeps_different_content(self):
        """All unique content chunks are kept."""
        contents = {
            "chunk_a": "Content about Python programming",
            "chunk_b": "Discussion of machine learning",
            "chunk_c": "Guide to web development",
        }

        def get_content(chunk_id: str):
            return contents.get(chunk_id)

        results = [
            ("chunk_a", 0.9),
            ("chunk_b", 0.8),
            ("chunk_c", 0.7),
        ]

        deduped, removed = deduplicate_by_ngram(results, get_content, threshold=0.7)

        assert len(deduped) == 3
        assert removed == 0

    def test_handles_missing_content_gracefully(self):
        """Chunks with None content are kept."""
        contents = {
            "chunk_a": "Hello world",
            # chunk_b returns None
            "chunk_c": "Hello world",
        }

        def get_content(chunk_id: str):
            return contents.get(chunk_id)

        results = [
            ("chunk_a", 0.9),
            ("chunk_b", 0.8),
            ("chunk_c", 0.7),
        ]

        deduped, removed = deduplicate_by_ngram(results, get_content, threshold=0.7)

        # chunk_b kept (None content), chunk_c removed (similar to chunk_a)
        assert len(deduped) == 2
        assert ("chunk_a", 0.9) in deduped
        assert ("chunk_b", 0.8) in deduped
        assert removed == 1

    def test_default_threshold_is_07(self):
        """Default threshold is 0.7."""
        contents = {
            "chunk_a": "The quick brown fox",
            "chunk_b": "The quick brown cat",  # Similar but not identical
        }

        def get_content(chunk_id: str):
            return contents.get(chunk_id)

        results = [
            ("chunk_a", 0.9),
            ("chunk_b", 0.8),
        ]

        # Use default threshold
        deduped, removed = deduplicate_by_ngram(results, get_content)

        # Should be deduplicated at default 0.7 threshold
        assert len(deduped) == 1
        assert removed == 1


class TestCosineSimilarity:
    """Tests for cosine_similarity function."""

    def test_identical_vectors_returns_one(self):
        """Identical vectors have similarity 1.0."""
        vec = np.array([1.0, 2.0, 3.0])
        similarity = cosine_similarity(vec, vec)
        assert similarity == pytest.approx(1.0)

    def test_orthogonal_vectors_returns_zero(self):
        """Orthogonal vectors have similarity 0.0."""
        vec_a = np.array([1.0, 0.0, 0.0])
        vec_b = np.array([0.0, 1.0, 0.0])
        similarity = cosine_similarity(vec_a, vec_b)
        assert similarity == pytest.approx(0.0)

    def test_opposite_vectors_returns_negative_one(self):
        """Opposite vectors have similarity -1.0."""
        vec_a = np.array([1.0, 0.0, 0.0])
        vec_b = np.array([-1.0, 0.0, 0.0])
        similarity = cosine_similarity(vec_a, vec_b)
        assert similarity == pytest.approx(-1.0)

    def test_known_similarity_value(self):
        """Test with known similarity calculation."""
        # vec_a = [1, 2, 3], vec_b = [4, 5, 6]
        # dot = 4 + 10 + 18 = 32
        # norm_a = sqrt(14) = 3.7417
        # norm_b = sqrt(77) = 8.7750
        # similarity = 32 / (3.7417 * 8.7750) = 0.9746
        vec_a = np.array([1.0, 2.0, 3.0])
        vec_b = np.array([4.0, 5.0, 6.0])
        similarity = cosine_similarity(vec_a, vec_b)
        assert similarity == pytest.approx(0.9746, abs=0.001)

    def test_zero_vector_returns_zero(self):
        """Zero vector returns similarity 0.0 (division by zero handled)."""
        vec_a = np.array([1.0, 2.0, 3.0])
        vec_b = np.array([0.0, 0.0, 0.0])
        similarity = cosine_similarity(vec_a, vec_b)
        assert similarity == 0.0

    def test_both_zero_vectors_returns_zero(self):
        """Two zero vectors return similarity 0.0."""
        vec_a = np.array([0.0, 0.0, 0.0])
        vec_b = np.array([0.0, 0.0, 0.0])
        similarity = cosine_similarity(vec_a, vec_b)
        assert similarity == 0.0

    def test_high_dimensional_vectors(self):
        """Works with high-dimensional vectors (typical embedding size)."""
        np.random.seed(42)
        vec_a = np.random.randn(384)  # BAAI/bge-small-en-v1.5 dimension
        vec_b = np.random.randn(384)

        similarity = cosine_similarity(vec_a, vec_b)

        # Should be a valid similarity between -1 and 1
        assert -1.0 <= similarity <= 1.0


class TestDeduplicateBySimilarity:
    """Tests for deduplicate_by_similarity function."""

    def test_empty_list_returns_empty(self):
        """Empty input returns empty list and 0 clusters merged."""
        def get_embedding(chunk_id: str):
            return None

        result, clusters_merged = deduplicate_by_similarity([], get_embedding)

        assert result == []
        assert clusters_merged == 0

    def test_single_item_returns_unchanged(self):
        """Single item returns unchanged, 0 clusters merged."""
        def get_embedding(chunk_id: str):
            return [0.1, 0.2, 0.3]

        results = [("chunk_a", 0.9)]
        deduped, clusters_merged = deduplicate_by_similarity(results, get_embedding)

        assert deduped == [("chunk_a", 0.9)]
        assert clusters_merged == 0

    def test_removes_duplicates_above_threshold(self):
        """Removes chunks with similarity above threshold."""
        embeddings = {
            "chunk_a": [1.0, 0.0, 0.0],
            "chunk_b": [0.99, 0.1, 0.0],  # Very similar to chunk_a
            "chunk_c": [0.0, 1.0, 0.0],   # Different from both
        }

        def get_embedding(chunk_id: str):
            return embeddings.get(chunk_id)

        results = [
            ("chunk_a", 0.9),
            ("chunk_b", 0.8),  # Should be removed (similar to chunk_a)
            ("chunk_c", 0.7),
        ]

        deduped, clusters_merged = deduplicate_by_similarity(
            results, get_embedding, similarity_threshold=0.9
        )

        # chunk_b removed as duplicate of chunk_a
        assert len(deduped) == 2
        assert ("chunk_a", 0.9) in deduped
        assert ("chunk_c", 0.7) in deduped
        assert ("chunk_b", 0.8) not in deduped
        assert clusters_merged == 1

    def test_keeps_highest_scored_representative(self):
        """First occurrence (highest score) is kept as representative."""
        embeddings = {
            "chunk_a": [1.0, 0.0, 0.0],
            "chunk_b": [1.0, 0.0, 0.0],  # Identical to chunk_a
        }

        def get_embedding(chunk_id: str):
            return embeddings.get(chunk_id)

        # Results already sorted by score (descending)
        results = [
            ("chunk_a", 0.9),  # Higher score, should be kept
            ("chunk_b", 0.7),  # Lower score, should be removed
        ]

        deduped, clusters_merged = deduplicate_by_similarity(
            results, get_embedding, similarity_threshold=0.85
        )

        assert deduped == [("chunk_a", 0.9)]
        assert clusters_merged == 1

    def test_handles_missing_embeddings_gracefully(self):
        """Chunks with missing embeddings are kept (not removed as duplicates)."""
        embeddings = {
            "chunk_a": [1.0, 0.0, 0.0],
            # chunk_b has no embedding
            "chunk_c": [0.0, 1.0, 0.0],
        }

        def get_embedding(chunk_id: str):
            return embeddings.get(chunk_id)

        results = [
            ("chunk_a", 0.9),
            ("chunk_b", 0.8),  # No embedding, should be kept
            ("chunk_c", 0.7),
        ]

        deduped, clusters_merged = deduplicate_by_similarity(
            results, get_embedding, similarity_threshold=0.85
        )

        # All chunks kept (no duplicates detected, chunk_b has no embedding)
        assert len(deduped) == 3
        assert clusters_merged == 0

    def test_threshold_one_keeps_all(self):
        """Threshold 1.0 only removes exact duplicates."""
        embeddings = {
            "chunk_a": [1.0, 0.0, 0.0],
            "chunk_b": [0.99, 0.1, 0.0],  # Very similar but not identical
        }

        def get_embedding(chunk_id: str):
            return embeddings.get(chunk_id)

        results = [
            ("chunk_a", 0.9),
            ("chunk_b", 0.8),
        ]

        deduped, clusters_merged = deduplicate_by_similarity(
            results, get_embedding, similarity_threshold=1.0
        )

        # Both kept (not exactly identical)
        assert len(deduped) == 2
        assert clusters_merged == 0

    def test_low_threshold_removes_most(self):
        """Low threshold removes more duplicates."""
        embeddings = {
            "chunk_a": [1.0, 0.0, 0.0],
            "chunk_b": [0.8, 0.6, 0.0],   # Somewhat similar
            "chunk_c": [0.1, 0.9, 0.4],   # Different
        }

        def get_embedding(chunk_id: str):
            return embeddings.get(chunk_id)

        results = [
            ("chunk_a", 0.9),
            ("chunk_b", 0.8),
            ("chunk_c", 0.7),
        ]

        deduped, clusters_merged = deduplicate_by_similarity(
            results, get_embedding, similarity_threshold=0.5
        )

        # chunk_b removed (similarity with chunk_a > 0.5)
        assert ("chunk_a", 0.9) in deduped
        assert clusters_merged >= 1

    def test_multiple_clusters_merged(self):
        """Correctly counts multiple clusters being merged."""
        embeddings = {
            "chunk_a": [1.0, 0.0, 0.0],
            "chunk_b": [1.0, 0.0, 0.0],   # Duplicate of a
            "chunk_c": [1.0, 0.0, 0.0],   # Duplicate of a
            "chunk_d": [0.0, 1.0, 0.0],   # Different
        }

        def get_embedding(chunk_id: str):
            return embeddings.get(chunk_id)

        results = [
            ("chunk_a", 0.9),
            ("chunk_b", 0.8),
            ("chunk_c", 0.7),
            ("chunk_d", 0.6),
        ]

        deduped, clusters_merged = deduplicate_by_similarity(
            results, get_embedding, similarity_threshold=0.99
        )

        assert len(deduped) == 2
        assert ("chunk_a", 0.9) in deduped
        assert ("chunk_d", 0.6) in deduped
        assert clusters_merged == 2  # chunk_b and chunk_c merged into chunk_a's cluster

    def test_default_threshold_is_085(self):
        """Default threshold is 0.85."""
        embeddings = {
            "chunk_a": [1.0, 0.0, 0.0],
            "chunk_b": [0.9, 0.44, 0.0],  # cosine ~0.9 > 0.85
        }

        def get_embedding(chunk_id: str):
            return embeddings.get(chunk_id)

        results = [
            ("chunk_a", 0.9),
            ("chunk_b", 0.8),
        ]

        # Use default threshold
        deduped, clusters_merged = deduplicate_by_similarity(results, get_embedding)

        # chunk_b should be removed at default threshold 0.85
        assert len(deduped) == 1
        assert clusters_merged == 1


class TestDeduplicateByContentHash:
    """Tests for deduplicate_by_content_hash function."""

    def test_empty_list_returns_empty(self):
        """Empty input returns empty list and 0 removed."""
        def get_content(chunk_id: str):
            return None

        result, removed = deduplicate_by_content_hash([], get_content)

        assert result == []
        assert removed == 0

    def test_single_item_returns_unchanged(self):
        """Single item returns unchanged, 0 removed."""
        def get_content(chunk_id: str):
            return "some content"

        results = [("chunk_a", 0.9)]
        deduped, removed = deduplicate_by_content_hash(results, get_content)

        assert deduped == [("chunk_a", 0.9)]
        assert removed == 0

    def test_removes_exact_duplicates(self):
        """Removes chunks with identical content."""
        contents = {
            "chunk_a": "Hello world",
            "chunk_b": "Hello world",  # Exact duplicate
            "chunk_c": "Different content",
        }

        def get_content(chunk_id: str):
            return contents.get(chunk_id)

        results = [
            ("chunk_a", 0.9),
            ("chunk_b", 0.8),
            ("chunk_c", 0.7),
        ]

        deduped, removed = deduplicate_by_content_hash(results, get_content)

        assert len(deduped) == 2
        assert ("chunk_a", 0.9) in deduped
        assert ("chunk_c", 0.7) in deduped
        assert ("chunk_b", 0.8) not in deduped
        assert removed == 1

    def test_keeps_first_occurrence(self):
        """First occurrence (highest score) is kept."""
        contents = {
            "chunk_a": "Same content",
            "chunk_b": "Same content",
        }

        def get_content(chunk_id: str):
            return contents.get(chunk_id)

        results = [
            ("chunk_a", 0.9),
            ("chunk_b", 0.7),
        ]

        deduped, removed = deduplicate_by_content_hash(results, get_content)

        assert deduped == [("chunk_a", 0.9)]
        assert removed == 1

    def test_handles_missing_content_gracefully(self):
        """Chunks with None content are kept (not removed as duplicates)."""
        contents = {
            "chunk_a": "Hello world",
            # chunk_b returns None
            "chunk_c": "Hello world",
        }

        def get_content(chunk_id: str):
            return contents.get(chunk_id)

        results = [
            ("chunk_a", 0.9),
            ("chunk_b", 0.8),
            ("chunk_c", 0.7),
        ]

        deduped, removed = deduplicate_by_content_hash(results, get_content)

        # chunk_b kept (None content), chunk_c removed (duplicate of chunk_a)
        assert len(deduped) == 2
        assert ("chunk_a", 0.9) in deduped
        assert ("chunk_b", 0.8) in deduped
        assert removed == 1

    def test_strips_whitespace_before_hashing(self):
        """Whitespace differences are ignored."""
        contents = {
            "chunk_a": "  Hello world  ",
            "chunk_b": "Hello world",
            "chunk_c": "\n\tHello world\n",
        }

        def get_content(chunk_id: str):
            return contents.get(chunk_id)

        results = [
            ("chunk_a", 0.9),
            ("chunk_b", 0.8),
            ("chunk_c", 0.7),
        ]

        deduped, removed = deduplicate_by_content_hash(results, get_content)

        # All three have same stripped content
        assert len(deduped) == 1
        assert deduped[0][0] == "chunk_a"
        assert removed == 2

    def test_multiple_duplicates_removed(self):
        """Correctly counts multiple duplicates being removed."""
        contents = {
            "chunk_a": "Content A",
            "chunk_b": "Content A",
            "chunk_c": "Content A",
            "chunk_d": "Content B",
        }

        def get_content(chunk_id: str):
            return contents.get(chunk_id)

        results = [
            ("chunk_a", 0.9),
            ("chunk_b", 0.8),
            ("chunk_c", 0.7),
            ("chunk_d", 0.6),
        ]

        deduped, removed = deduplicate_by_content_hash(results, get_content)

        assert len(deduped) == 2
        assert ("chunk_a", 0.9) in deduped
        assert ("chunk_d", 0.6) in deduped
        assert removed == 2

    def test_different_content_all_kept(self):
        """All unique content chunks are kept."""
        contents = {
            "chunk_a": "Content A",
            "chunk_b": "Content B",
            "chunk_c": "Content C",
        }

        def get_content(chunk_id: str):
            return contents.get(chunk_id)

        results = [
            ("chunk_a", 0.9),
            ("chunk_b", 0.8),
            ("chunk_c", 0.7),
        ]

        deduped, removed = deduplicate_by_content_hash(results, get_content)

        assert len(deduped) == 3
        assert removed == 0
