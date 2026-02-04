"""
Unit tests for SearchOrchestrator embedding cache thread safety.

Tests verify that concurrent access to the embedding cache does not
cause race conditions, KeyError exceptions, or cache corruption.
"""

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.config import ChunkingConfig, Config, IndexingConfig, LLMConfig, SearchConfig
from src.indices.graph import GraphStore
from src.indices.keyword import KeywordIndex
from src.indices.vector import VectorIndex
from src.search.orchestrator import SearchOrchestrator


@pytest.fixture
def config(tmp_path):
    docs_path = tmp_path / "docs"
    docs_path.mkdir()
    index_path = tmp_path / "indices"
    index_path.mkdir()
    return Config(
        indexing=IndexingConfig(
            documents_path=str(docs_path),
            index_path=str(index_path),
        ),
        search=SearchConfig(
            semantic_weight=1.0,
            keyword_weight=1.0,
            rrf_k_constant=60,
        ),
        llm=LLMConfig(embedding_model="BAAI/bge-small-en-v1.5"),
        document_chunking=ChunkingConfig(),
        memory_chunking=ChunkingConfig(),
    )


@pytest.fixture
def mock_vector_index():
    """Create a mock VectorIndex that returns deterministic embeddings."""
    mock = MagicMock(spec=VectorIndex)
    # Return a unique embedding per query (based on hash)
    mock.get_text_embedding.side_effect = lambda q: [float(hash(q) % 1000) / 1000.0] * 384
    return mock


@pytest.fixture
def orchestrator(config, mock_vector_index):
    """Create orchestrator with mocked vector index for cache testing."""
    keyword = KeywordIndex()
    graph = GraphStore()
    orch = SearchOrchestrator(
        vector_index=mock_vector_index,
        keyword_index=keyword,
        graph_store=graph,
        config=config,
        index_manager=None,
        documents_path=Path(config.indexing.documents_path),
    )
    # Reduce cache size to trigger eviction faster
    orch._cache_max_size = 10
    return orch


class TestEmbeddingCacheThreadSafety:
    """Tests for concurrent cache access."""

    def test_concurrent_cache_access_no_exceptions(self, orchestrator, mock_vector_index):
        """
        Test that concurrent cache access from multiple threads does not raise exceptions.

        Simulates multiple threads accessing _get_cached_embedding simultaneously
        with different queries, triggering cache reads, writes, and evictions.
        """
        queries = [f"query_{i}" for i in range(50)]
        results = []
        exceptions = []

        def get_embedding(query: str):
            try:
                return orchestrator._get_cached_embedding(query)
            except Exception as e:
                exceptions.append(e)
                return None

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(get_embedding, q) for q in queries]
            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    results.append(result)

        assert len(exceptions) == 0, f"Exceptions raised during concurrent access: {exceptions}"
        assert len(results) == len(queries)

    def test_concurrent_same_query_access(self, orchestrator, mock_vector_index):
        """
        Test that concurrent access for the same query is handled correctly.

        Multiple threads requesting the same query should all succeed,
        with only one actual embedding computation (cache hit for others).
        """
        query = "shared_query"
        num_threads = 20
        results = []
        exceptions = []

        def get_embedding():
            try:
                return orchestrator._get_cached_embedding(query)
            except Exception as e:
                exceptions.append(e)
                return None

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(get_embedding) for _ in range(num_threads)]
            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    results.append(result)

        assert len(exceptions) == 0, f"Exceptions raised: {exceptions}"
        assert len(results) == num_threads
        # All results should be identical (same embedding for same query)
        assert all(r == results[0] for r in results)

    def test_concurrent_eviction_no_keyerror(self, orchestrator, mock_vector_index):
        """
        Test that concurrent eviction does not cause KeyError.

        When cache is full and multiple threads try to add entries,
        eviction should be handled safely without race conditions.
        """
        # Fill cache to capacity
        for i in range(orchestrator._cache_max_size):
            orchestrator._get_cached_embedding(f"prefill_{i}")

        # Now trigger concurrent evictions
        new_queries = [f"new_query_{i}" for i in range(30)]
        exceptions = []

        def get_embedding(query: str):
            try:
                return orchestrator._get_cached_embedding(query)
            except Exception as e:
                exceptions.append(e)
                return None

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(get_embedding, q) for q in new_queries]
            for future in as_completed(futures):
                future.result()

        assert len(exceptions) == 0, f"Exceptions raised during eviction: {exceptions}"

    def test_cache_lock_exists(self, orchestrator):
        """Verify that the orchestrator has a cache lock attribute."""
        assert hasattr(orchestrator, "_cache_lock")
        assert orchestrator._cache_lock is not None

    def test_cache_hit_returns_same_embedding(self, orchestrator, mock_vector_index):
        """Test that cache hits return the cached embedding without recomputation."""
        query = "test_cache_hit"

        # First call - cache miss
        embedding1 = orchestrator._get_cached_embedding(query)
        call_count_after_first = mock_vector_index.get_text_embedding.call_count

        # Second call - should be cache hit
        embedding2 = orchestrator._get_cached_embedding(query)
        call_count_after_second = mock_vector_index.get_text_embedding.call_count

        assert embedding1 == embedding2
        assert call_count_after_second == call_count_after_first  # No new computation

    def test_expired_entry_triggers_recomputation(self, orchestrator, mock_vector_index):
        """Test that expired cache entries are recomputed."""
        query = "test_expiry"
        orchestrator._cache_ttl = 0.1  # 100ms TTL for test

        # First call
        orchestrator._get_cached_embedding(query)
        call_count_after_first = mock_vector_index.get_text_embedding.call_count

        # Wait for expiry
        time.sleep(0.15)

        # Second call should recompute (expired)
        orchestrator._get_cached_embedding(query)
        call_count_after_second = mock_vector_index.get_text_embedding.call_count

        assert call_count_after_second == call_count_after_first + 1
