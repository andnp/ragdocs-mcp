"""
Integration tests for query expansion via embeddings.

Tests cover:
- Building concept vocabulary from empty/populated index
- Term extraction and stopword filtering
- Query expansion with nearest terms
- Duplicate term handling
- Vocabulary persistence

Test strategies:
- Use real embedding model (shared_embedding_model from conftest)
- Test vocabulary building with known document content
- Verify expansion behavior with real embeddings
"""

from datetime import datetime

import pytest

from src.indices.vector import VectorIndex, STOPWORDS
from src.models import Chunk


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def vector_index_empty(shared_embedding_model):
    """
    Create empty VectorIndex with shared embedding model.
    """
    return VectorIndex(embedding_model=shared_embedding_model)


@pytest.fixture
def vector_index_populated(shared_embedding_model):
    """
    Create VectorIndex populated with test chunks.

    Contains chunks about authentication, authorization, and login.
    """
    index = VectorIndex(embedding_model=shared_embedding_model)

    chunks = [
        Chunk(
            chunk_id="doc1_chunk_0",
            doc_id="doc1",
            content="Authentication is the process of verifying user identity.",
            metadata={"tags": [], "links": []},
            chunk_index=0,
            header_path="Security > Authentication",
            start_pos=0,
            end_pos=60,
            file_path="/docs/auth.md",
            modified_time=datetime.now(),
        ),
        Chunk(
            chunk_id="doc2_chunk_0",
            doc_id="doc2",
            content="Authorization determines what a user can access after login.",
            metadata={"tags": [], "links": []},
            chunk_index=0,
            header_path="Security > Authorization",
            start_pos=0,
            end_pos=60,
            file_path="/docs/authz.md",
            modified_time=datetime.now(),
        ),
        Chunk(
            chunk_id="doc3_chunk_0",
            doc_id="doc3",
            content="The login page allows users to enter credentials for authentication.",
            metadata={"tags": [], "links": []},
            chunk_index=0,
            header_path="User Interface > Login",
            start_pos=0,
            end_pos=70,
            file_path="/docs/login.md",
            modified_time=datetime.now(),
        ),
    ]

    for chunk in chunks:
        index.add_chunk(chunk)

    return index


# ============================================================================
# Build Vocabulary - Empty Index Tests
# ============================================================================


class TestBuildVocabularyEmptyIndex:
    """Tests for vocabulary building with empty index."""

    def test_build_vocabulary_empty_index_produces_empty_vocab(self, vector_index_empty):
        """
        Empty index produces empty vocabulary.

        No documents means no terms to extract.
        """
        vector_index_empty.build_concept_vocabulary()

        assert vector_index_empty._concept_vocabulary == {}

    def test_build_vocabulary_empty_index_no_errors(self, vector_index_empty):
        """
        Building vocabulary on empty index doesn't raise errors.

        Graceful handling of edge case.
        """
        # Should not raise
        vector_index_empty.build_concept_vocabulary()


# ============================================================================
# Build Vocabulary - Term Extraction Tests
# ============================================================================


class TestBuildVocabularyExtractsTerms:
    """Tests for term extraction from documents."""

    def test_build_vocabulary_extracts_terms(self, vector_index_populated):
        """
        Extracts terms from indexed documents.

        Vocabulary should contain terms from chunk content.
        """
        vector_index_populated.build_concept_vocabulary()

        vocab = vector_index_populated._concept_vocabulary
        assert len(vocab) > 0

        # Should contain key terms from content
        vocab_terms = set(vocab.keys())
        assert "authentication" in vocab_terms
        assert "authorization" in vocab_terms
        assert "login" in vocab_terms
        assert "user" in vocab_terms

    def test_build_vocabulary_respects_min_term_length(self, vector_index_populated):
        """
        Filters out terms shorter than min_term_length.

        Short terms like "a", "is" should be excluded.
        """
        vector_index_populated.build_concept_vocabulary(min_term_length=4)

        vocab_terms = set(vector_index_populated._concept_vocabulary.keys())

        # No terms shorter than 4 chars
        for term in vocab_terms:
            assert len(term) >= 4

    def test_build_vocabulary_respects_max_terms(self, vector_index_populated):
        """
        Limits vocabulary to max_terms most frequent terms.

        Tests vocabulary size constraint.
        """
        vector_index_populated.build_concept_vocabulary(max_terms=5)

        assert len(vector_index_populated._concept_vocabulary) <= 5


# ============================================================================
# Build Vocabulary - Stopword Filtering Tests
# ============================================================================


class TestBuildVocabularyFiltersStopwords:
    """Tests for stopword filtering in vocabulary building."""

    def test_build_vocabulary_filters_stopwords(self, vector_index_populated):
        """
        Stopwords not included in vocabulary.

        Common words like "the", "is", "a" should be excluded.
        """
        vector_index_populated.build_concept_vocabulary()

        vocab_terms = set(vector_index_populated._concept_vocabulary.keys())

        # None of the stopwords should be in vocabulary
        for stopword in ["the", "is", "a", "of", "to", "and", "for"]:
            assert stopword not in vocab_terms

    def test_stopwords_constant_includes_common_words(self):
        """
        STOPWORDS constant includes common English words.

        Verifies the stopword list is properly defined.
        """
        common_stopwords = {"the", "a", "an", "is", "are", "was", "were", "be"}
        assert common_stopwords.issubset(STOPWORDS)


# ============================================================================
# Query Expansion - Basic Tests
# ============================================================================


class TestExpandQueryAddsTerms:
    """Tests for basic query expansion behavior."""

    def test_expand_query_adds_terms(self, vector_index_populated):
        """
        Expansion adds related terms to query.

        Query should be augmented with semantically similar terms.
        """
        vector_index_populated.build_concept_vocabulary()

        original_query = "auth"
        expanded_query = vector_index_populated.expand_query(
            original_query,
            top_k=3,
            similarity_threshold=0.0,  # Low threshold to ensure expansion
        )

        # Expanded query should be longer
        assert len(expanded_query) > len(original_query)
        # Original query terms preserved
        assert "auth" in expanded_query.lower()

    def test_expand_query_returns_string(self, vector_index_populated):
        """
        Expanded query is a string.

        Type verification for API contract.
        """
        vector_index_populated.build_concept_vocabulary()

        result = vector_index_populated.expand_query("authentication")

        assert isinstance(result, str)


# ============================================================================
# Query Expansion - No Duplicate Terms Tests
# ============================================================================


class TestExpandQueryNoDuplicateTerms:
    """Tests for duplicate term handling in expansion."""

    def test_expand_query_no_duplicate_terms(self, vector_index_populated):
        """
        Terms already in query not added again.

        Prevents redundant expansion.
        """
        vector_index_populated.build_concept_vocabulary()

        # Query already contains "authentication"
        original_query = "authentication process"
        expanded_query = vector_index_populated.expand_query(
            original_query,
            top_k=5,
            similarity_threshold=0.0,
        )

        # Count occurrences of "authentication"
        tokens = expanded_query.lower().split()
        auth_count = tokens.count("authentication")

        # Should only appear once (from original query)
        assert auth_count <= 1


# ============================================================================
# Query Expansion - Empty Vocabulary Tests
# ============================================================================


class TestExpandQueryEmptyVocabulary:
    """Tests for expansion with empty vocabulary."""

    def test_expand_query_empty_vocabulary_returns_original(self, vector_index_empty):
        """
        Returns original query if no vocabulary.

        Graceful degradation when vocabulary not built.
        """
        # Don't build vocabulary
        original_query = "test query"
        result = vector_index_empty.expand_query(original_query)

        assert result == original_query

    def test_expand_query_empty_vocabulary_no_errors(self, vector_index_empty):
        """
        No errors when expanding with empty vocabulary.

        Safe fallback behavior.
        """
        # Should not raise
        result = vector_index_empty.expand_query("test query")
        assert isinstance(result, str)


# ============================================================================
# Vocabulary Persistence Tests
# ============================================================================


class TestVocabularyPersistence:
    """Tests for vocabulary save and load."""

    def test_vocabulary_persistence_saved(self, vector_index_populated, tmp_path):
        """
        Vocabulary saved to disk during persist.

        Verifies file is created.
        """
        vector_index_populated.build_concept_vocabulary()
        persist_path = tmp_path / "index"

        vector_index_populated.persist(persist_path)

        vocab_file = persist_path / "concept_vocabulary.json"
        assert vocab_file.exists()

    def test_vocabulary_persistence_loaded(self, shared_embedding_model, tmp_path):
        """
        Vocabulary loaded from disk correctly.

        Full round-trip test: save and load.
        """
        # Create and populate first index
        index1 = VectorIndex(embedding_model=shared_embedding_model)
        chunk = Chunk(
            chunk_id="doc1_chunk_0",
            doc_id="doc1",
            content="Authentication security credentials verification process.",
            metadata={},
            chunk_index=0,
            header_path="",
            start_pos=0,
            end_pos=60,
            file_path="/docs/test.md",
            modified_time=datetime.now(),
        )
        index1.add_chunk(chunk)
        index1.build_concept_vocabulary()

        persist_path = tmp_path / "index"
        index1.persist(persist_path)

        # Create second index and load
        index2 = VectorIndex(embedding_model=shared_embedding_model)
        index2.load(persist_path)

        # Vocabulary should be loaded
        assert len(index2._concept_vocabulary) > 0
        assert "authentication" in index2._concept_vocabulary or "security" in index2._concept_vocabulary

    def test_vocabulary_persistence_missing_file(self, shared_embedding_model, tmp_path):
        """
        Handles missing vocabulary file gracefully.

        When loading an index that was persisted without a vocabulary,
        the vocabulary should remain empty.
        """
        persist_path = tmp_path / "index"

        # Create a real index and persist it WITHOUT building vocabulary
        index1 = VectorIndex(embedding_model=shared_embedding_model)
        chunk = Chunk(
            chunk_id="doc1_chunk_0",
            doc_id="doc1",
            content="Test content for index.",
            metadata={},
            chunk_index=0,
            header_path="",
            start_pos=0,
            end_pos=30,
            file_path="/docs/test.md",
            modified_time=datetime.now(),
        )
        index1.add_chunk(chunk)
        # Note: NOT calling build_concept_vocabulary()
        index1.persist(persist_path)

        # Now delete the vocabulary file if it exists
        vocab_file = persist_path / "concept_vocabulary.json"
        if vocab_file.exists():
            vocab_file.unlink()

        # Load into new index - should handle missing vocab file
        index2 = VectorIndex(embedding_model=shared_embedding_model)
        index2.load(persist_path)

        # Should have empty vocabulary (not crash)
        assert index2._concept_vocabulary == {}


# ============================================================================
# Query Expansion - Similarity Threshold Tests
# ============================================================================


class TestExpandQuerySimilarityThreshold:
    """Tests for similarity threshold in expansion."""

    def test_expand_query_high_threshold_fewer_terms(self, vector_index_populated):
        """
        High similarity threshold results in fewer expansion terms.

        Stricter matching = less expansion.
        """
        vector_index_populated.build_concept_vocabulary()

        query = "user"

        expanded_low = vector_index_populated.expand_query(
            query,
            top_k=5,
            similarity_threshold=0.0,
        )
        expanded_high = vector_index_populated.expand_query(
            query,
            top_k=5,
            similarity_threshold=0.9,
        )

        # High threshold should have fewer or equal terms
        assert len(expanded_high.split()) <= len(expanded_low.split())

    def test_expand_query_threshold_one_no_expansion(self, vector_index_populated):
        """
        Threshold of 1.0 means no expansion (impossible to match).

        Only exact embedding match would pass.
        """
        vector_index_populated.build_concept_vocabulary()

        query = "test query"
        expanded = vector_index_populated.expand_query(
            query,
            top_k=5,
            similarity_threshold=1.0,
        )

        # Should return original query unchanged
        assert expanded == query


# ============================================================================
# Query Expansion - top_k Tests
# ============================================================================


class TestExpandQueryTopK:
    """Tests for top_k parameter in expansion."""

    def test_expand_query_respects_top_k(self, vector_index_populated):
        """
        Expansion limited to top_k terms.

        At most top_k new terms added.
        """
        vector_index_populated.build_concept_vocabulary()

        query = "security"
        top_k = 2

        expanded = vector_index_populated.expand_query(
            query,
            top_k=top_k,
            similarity_threshold=0.0,
        )

        # Count new terms (expanded - original)
        original_tokens = set(query.lower().split())
        expanded_tokens = set(expanded.lower().split())
        new_tokens = expanded_tokens - original_tokens

        assert len(new_tokens) <= top_k

    def test_expand_query_top_k_zero_no_expansion(self, vector_index_populated):
        """
        top_k=0 means no expansion terms added.
        """
        vector_index_populated.build_concept_vocabulary()

        query = "test"
        expanded = vector_index_populated.expand_query(
            query,
            top_k=0,
            similarity_threshold=0.0,
        )

        assert expanded == query
