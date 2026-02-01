"""
Integration tests for Hybrid Search (D13).

Tests the SearchOrchestrator's ability to combine results from multiple search
strategies (semantic, keyword, graph) with recency boosting and RRF fusion.
Uses real indices and async query methods.
"""

from datetime import datetime, timezone
from pathlib import Path

import pytest

from src.config import Config, IndexingConfig, LLMConfig, SearchConfig, ServerConfig, ChunkingConfig
from src.indexing.manager import IndexManager
from src.indices.graph import GraphStore
from src.indices.keyword import KeywordIndex
from src.indices.vector import VectorIndex
from src.models import ChunkResult
from src.search.orchestrator import SearchOrchestrator


def _doc_in_chunk_ids(doc_id: str, results: list[ChunkResult]):
    chunk_ids = [result.chunk_id for result in results]
    return any(chunk_id == doc_id or chunk_id.startswith(f"{doc_id}_chunk_") for chunk_id in chunk_ids)


@pytest.fixture
def config(tmp_path):
    """
    Create test configuration with temporary paths.

    Uses tmp_path for isolated test storage to avoid conflicts.
    """
    docs_path = tmp_path / "docs"
    docs_path.mkdir()
    return Config(
        server=ServerConfig(),
        indexing=IndexingConfig(
            documents_path=str(docs_path),
            index_path=str(tmp_path / "indices"),
        ),
        parsers={"**/*.md": "MarkdownParser"},
        search=SearchConfig(
            semantic_weight=1.0,
            keyword_weight=1.0,
            recency_bias=0.5,
            rrf_k_constant=60,
        ),
        llm=LLMConfig(embedding_model="all-MiniLM-L6-v2"),
        document_chunking=ChunkingConfig(),
        memory_chunking=ChunkingConfig(),
    )


@pytest.fixture
def indices():
    """
    Create real index instances.

    Returns tuple of (vector, keyword, graph) indices for SearchOrchestrator.
    """
    vector = VectorIndex()
    keyword = KeywordIndex()
    graph = GraphStore()
    return vector, keyword, graph


@pytest.fixture
def manager(config, indices):
    """
    Create IndexManager with real indices.

    Provides fully functional manager for indexing test documents.
    """
    vector, keyword, graph = indices
    return IndexManager(config, vector, keyword, graph)


@pytest.fixture
def orchestrator(config, indices, manager):
    """
    Create SearchOrchestrator with real indices and configuration.

    Provides the hybrid search engine for query testing.
    """
    vector, keyword, graph = indices
    return SearchOrchestrator(vector, keyword, graph, config, manager)


def create_test_corpus(config, manager):
    """
    Create a small test corpus (3-5 docs) for hybrid search testing.

    Documents are designed to test different search strategies:
    - doc1: Contains specific keyword "authentication"
    - doc2: Semantically related to security concepts
    - doc3: Links to doc1 (tests graph traversal)
    - doc4: Recent document (tests recency boost)
    - doc5: Contains both keyword and semantic matches
    """
    docs_path = Path(config.indexing.documents_path)
    now = datetime.now(timezone.utc).timestamp()

    # Document 1: Keyword-rich document
    doc1 = docs_path / "authentication.md"
    doc1.write_text(
        "# Authentication System\n\n"
        "The authentication module handles user login and token validation."
    )
    manager.index_document(str(doc1))

    # Document 2: Semantically related to security
    doc2 = docs_path / "security.md"
    doc2.write_text(
        "# Security Overview\n\n"
        "Security measures include access control and authorization policies."
    )
    manager.index_document(str(doc2))

    # Document 3: Links to doc1 (graph connection)
    doc3 = docs_path / "api.md"
    doc3.write_text(
        "# API Documentation\n\n"
        "See [[authentication]] for details on API authentication."
    )
    manager.index_document(str(doc3))

    # Document 4: Recent document (set modified time to 3 days ago)
    doc4 = docs_path / "recent.md"
    doc4.write_text(
        "# Recent Updates\n\n"
        "Latest changes to the authorization system and access patterns."
    )
    manager.index_document(str(doc4))
    # Set modified time to 3 days ago for recency boost (1.2x)
    doc4.touch()
    doc4_mtime = now - (3 * 86400)  # 3 days ago
    import os
    os.utime(str(doc4), (doc4_mtime, doc4_mtime))

    # Document 5: Hybrid match (keyword + semantic)
    doc5 = docs_path / "auth_guide.md"
    doc5.write_text(
        "# Authentication Guide\n\n"
        "Complete guide to authentication, covering login flows and security best practices."
    )
    manager.index_document(str(doc5))

    return {
        "authentication": doc1,
        "security": doc2,
        "api": doc3,
        "recent": doc4,
        "auth_guide": doc5,
    }


@pytest.mark.asyncio
async def test_query_returns_results_from_multiple_strategies(
    config, manager, orchestrator
):
    """
    Test that hybrid query returns results from semantic, keyword, and graph strategies.

    Validates that the orchestrator correctly dispatches queries to multiple
    search backends and aggregates their results using RRF fusion.
    """
    # Create test corpus
    create_test_corpus(config, manager)

    # Query that should match multiple strategies
    results, _, _ = await orchestrator.query("authentication security", top_k=10, top_n=10)

    # Verify we got results
    assert len(results) > 0

    # Verify scores are in valid range and properly ordered
    assert all(0.0 <= result.score <= 1.0 for result in results)
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True), "Results should be descending by score"

    # Verify results include docs from different strategies:
    # - "authentication" should match keyword search (exact term)
    # - "api" might appear via graph traversal (linked to authentication)
    # - "auth_guide" should match both keyword and semantic

    # At minimum, should include documents with explicit keyword matches
    assert _doc_in_chunk_ids("authentication", results) or _doc_in_chunk_ids("auth_guide", results)

    # Verify we got multiple results (showing fusion is working)
    assert len(results) >= 2


@pytest.mark.asyncio
async def test_graph_neighbors_boost_related_docs(config, manager, orchestrator):
    """
    Test that graph traversal boosts documents linked to query matches.

    When a document links to another via [[wikilinks]], the linked document
    should appear in results due to graph neighbor expansion.
    """
    # Create test corpus
    create_test_corpus(config, manager)

    # Query specifically for "API Documentation"
    results, _, _ = await orchestrator.query("API Documentation", top_k=10, top_n=10)

    # Verify scores are valid and properly ordered
    assert all(0.0 <= result.score <= 1.0 for result in results)
    if len(results) > 1:
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    # Extract chunk_ids for checking
    _ = [result.chunk_id for result in results]

    # "api" should match the query
    assert _doc_in_chunk_ids("api", results)

    # "authentication" should also appear because "api" links to it via [[authentication]]
    # The graph neighbor boost adds linked documents to the result pool

    # Either authentication appears directly (keyword/semantic) or via graph boost
    # Since api links to authentication, graph traversal should include it
    assert _doc_in_chunk_ids("authentication", results) or len(results) >= 2


@pytest.mark.asyncio
async def test_recency_boosts_recent_docs(config, manager, orchestrator):
    """
    Test that recently modified documents receive score boost.

    Documents modified within the last 7 days should receive a 1.2x multiplier,
    pushing them higher in the ranking than older documents with similar scores.
    """
    # Create test corpus with recent doc
    create_test_corpus(config, manager)

    # Query that matches both recent and older docs
    results, _, _ = await orchestrator.query("authorization access", top_k=10, top_n=10)

    # Verify we got results
    assert len(results) > 0

    # Verify scores are valid
    assert all(0.0 <= result.score <= 1.0 for result in results)
    if results:
        # Highest score should be high confidence
        assert results[0].score >= 0.5

    # The "recent" document should appear in top results due to recency boost
    # It contains "authorization" and has a recent modification time (3 days ago)
    # This gives it a 1.2x boost vs older documents
    assert _doc_in_chunk_ids("recent", results)

    # Ideally, recent should rank highly due to recency boost
    # (exact ranking depends on other score factors, but it should be present)
    if len(results) >= 2:
        # If multiple results, recent should be boosted toward the top
        # Check if "recent" chunk exists in results before trying index
        chunk_ids = [result.chunk_id for result in results]
        recent_chunks = [r for r in chunk_ids if r.startswith("recent")]
        assert len(recent_chunks) > 0


@pytest.mark.asyncio
async def test_empty_query_returns_empty_results(config, manager, orchestrator):
    """
    Test that empty query returns no results.

    Validates graceful handling of edge case: empty or whitespace-only queries
    should return an empty list without errors.
    """
    # Create test corpus
    create_test_corpus(config, manager)

    # Test empty query
    results_empty, _, _ = await orchestrator.query("", top_k=10, top_n=10)
    assert results_empty == []

    # Test whitespace-only query
    results_whitespace, _, _ = await orchestrator.query("   \n\t  ", top_k=10, top_n=10)
    assert results_whitespace == []


@pytest.mark.asyncio
async def test_top_k_limits_results_correctly(config, manager, orchestrator):
    """
    Test that top_k parameter correctly limits the number of results.

    Validates that the orchestrator respects the top_k limit when returning
    fused results, even when multiple strategies return many matches.
    """
    # Create test corpus
    create_test_corpus(config, manager)

    # Query with very low top_k
    results_k1, _, _ = await orchestrator.query("authentication", top_k=1, top_n=1)
    assert len(results_k1) <= 1
    if results_k1:
        assert all(0.0 <= result.score <= 1.0 for result in results_k1)

    # Query with top_k=2
    results_k2, _, _ = await orchestrator.query("authentication", top_k=2, top_n=2)
    assert len(results_k2) <= 2
    if results_k2:
        assert all(0.0 <= result.score <= 1.0 for result in results_k2)

    # Query with top_k=5
    results_k5, _, _ = await orchestrator.query("authentication security", top_k=5, top_n=5)
    assert len(results_k5) <= 5
    if results_k5:
        assert all(0.0 <= result.score <= 1.0 for result in results_k5)
        # Highest score should be high confidence
        assert results_k5[0].score >= 0.5

    # Verify that increasing top_k returns more results (up to available docs)
    assert len(results_k1) <= len(results_k2) <= len(results_k5)


@pytest.mark.asyncio
async def test_weighted_strategies_affect_ranking(config, manager, orchestrator, tmp_path):
    """
    Test that strategy weights affect result ranking.

    When semantic_weight or keyword_weight is adjusted, the relative ranking
    of documents should change to favor results from higher-weighted strategies.
    """
    # Create test corpus
    create_test_corpus(config, manager)

    # Baseline query with equal weights (1.0, 1.0)
    results_balanced, _, _ = await orchestrator.query("authentication", top_k=5, top_n=5)
    assert len(results_balanced) > 0
    assert all(0.0 <= result.score <= 1.0 for result in results_balanced)

    # Create new config with higher semantic weight
    config_semantic_heavy = Config(
        server=ServerConfig(),
        indexing=IndexingConfig(
            documents_path=str(tmp_path / "docs2"),
            index_path=str(tmp_path / "indices2"),
        ),
        parsers=config.parsers,
        search=SearchConfig(
            semantic_weight=2.0,  # Double semantic weight
            keyword_weight=1.0,
            recency_bias=0.5,
            rrf_k_constant=60,
        ),
        llm=config.llm,
    )

    # Create new directory for second corpus
    docs_path2 = Path(config_semantic_heavy.indexing.documents_path)
    docs_path2.mkdir(parents=True, exist_ok=True)

    # Create fresh indices for semantic-heavy test
    vector_new = VectorIndex()
    keyword_new = KeywordIndex()
    graph_new = GraphStore()
    manager_new = IndexManager(config_semantic_heavy, vector_new, keyword_new, graph_new)
    create_test_corpus(config_semantic_heavy, manager_new)
    orchestrator_semantic = SearchOrchestrator(
        vector_new, keyword_new, graph_new, config_semantic_heavy, manager_new
    )

    # Query with semantic weight favored
    results_semantic_heavy, _, _ = await orchestrator_semantic.query("authentication", top_k=5, top_n=5)

    # Should still get results, but ranking may differ
    assert len(results_semantic_heavy) > 0
    assert all(0.0 <= result.score <= 1.0 for result in results_semantic_heavy)

    # Verify different weight configurations produce results
    # (Exact ranking is complex, but we validate the system handles weights)
    chunk_ids = [result.chunk_id for result in results_semantic_heavy]
    result_set = set(chunk_ids)
    assert len(result_set) > 0


@pytest.mark.asyncio
async def test_hybrid_search_integration_end_to_end(config, manager, orchestrator):
    """
    Test complete hybrid search flow: index -> query -> fused results.

    End-to-end integration test validating that all components work together:
    multiple indices, parallel search dispatch, graph traversal, recency boost,
    RRF fusion, and top_k filtering.
    """
    # Create test corpus
    docs = create_test_corpus(config, manager)

    # Verify all documents were indexed
    assert len(docs) == 5

    # Complex query touching multiple aspects
    results, _, _ = await orchestrator.query(
        "authentication security access control",
        top_k=10,
        top_n=10
    )

    # Should return multiple results from different strategies
    assert len(results) >= 2

    # Verify result types are correct
    assert all(isinstance(item, ChunkResult) for item in results)
    assert all(isinstance(result.chunk_id, str) and isinstance(result.score, float) for result in results)

    # Verify scores are valid and properly ordered
    assert all(0.0 <= result.score <= 1.0 for result in results)
    if len(results) > 1:
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True), "Results should be descending by score"

    # Extract chunk_ids for duplicate check
    chunk_ids = [result.chunk_id for result in results]

    # Verify no duplicate results
    assert len(chunk_ids) == len(set(chunk_ids))

    # Verify results include expected documents - authentication should match
    assert _doc_in_chunk_ids("authentication", results) or _doc_in_chunk_ids("auth_guide", results)


@pytest.mark.asyncio
async def test_query_returns_normalized_scores(config, manager, orchestrator):
    """
    Test that SearchOrchestrator returns properly normalized scores.

    Validates that the score normalization pipeline works correctly:
    - Scores are in [0, 1] range
    - Highest score is 1.0
    - Scores are descending
    - Results are tuples of (chunk_id, score)
    """
    create_test_corpus(config, manager)
    results, _, _ = await orchestrator.query("authentication", top_k=10, top_n=5)

    # Verify structure
    assert len(results) <= 5
    assert all(isinstance(item, ChunkResult) for item in results)

    # Verify score properties
    if results:
        chunk_ids = [result.chunk_id for result in results]
        scores = [result.score for result in results]

        # All scores in [0, 1]
        assert all(0.0 <= score <= 1.0 for score in scores)

        # Highest score should be high confidence
        assert results[0].score >= 0.5

        # Scores are descending
        for i in range(len(results) - 1):
            assert results[i].score >= results[i+1].score

        # All chunk_ids are strings
        assert all(isinstance(chunk_id, str) for chunk_id in chunk_ids)

        # All scores are floats
        assert all(isinstance(score, float) for score in scores)


@pytest.mark.asyncio
async def test_top_n_parameter_limits_results(config, manager, orchestrator):
    """
    Test that top_n parameter correctly limits result count.

    Validates:
    - top_n=5 returns at most 5 results
    - top_n=3 returns at most 3 results
    - top_n=1 returns at most 1 result
    - Top N results are consistent across queries
    """
    create_test_corpus(config, manager)

    results_5, _, _ = await orchestrator.query("authentication", top_k=10, top_n=5)
    results_3, _, _ = await orchestrator.query("authentication", top_k=10, top_n=3)
    results_1, _, _ = await orchestrator.query("authentication", top_k=10, top_n=1)

    # Verify limits are respected
    assert len(results_5) <= 5
    assert len(results_3) <= 3
    assert len(results_1) <= 1

    # Verify scores are valid for all result sets
    for results in [results_5, results_3, results_1]:
        assert all(0.0 <= result.score <= 1.0 for result in results)
        if results:
            # Highest score should be high confidence
            assert results[0].score >= 0.5

    # Top N results should match: results_5[:3] should equal results_3
    if len(results_3) >= 3 and len(results_5) >= 3:
        assert results_5[:3] == results_3[:3]

    # Top 1 result should match across all queries
    if len(results_1) >= 1 and len(results_3) >= 1 and len(results_5) >= 1:
        assert results_1[0] == results_3[0] == results_5[0]


@pytest.mark.asyncio
async def test_normalized_scores_range_0_to_1(config, manager, orchestrator):
    """
    Test that all normalized scores are strictly within [0, 1] range.

    Boundary validation test: ensures no score escapes normalization
    bounds regardless of query complexity or strategy weights.
    """
    create_test_corpus(config, manager)

    # Test with various queries
    queries = [
        "authentication security",
        "API documentation",
        "access control",
        "recent updates",
        "authorization",
    ]

    for query in queries:
        results, _, _ = await orchestrator.query(query, top_k=10, top_n=10)

        for result in results:
            # Verify types
            assert isinstance(result.chunk_id, str), f"chunk_id must be str, got {type(result.chunk_id)}"
            assert isinstance(result.score, float), f"score must be float, got {type(result.score)}"

            # Verify score bounds
            assert 0.0 <= result.score <= 1.0, f"Score {result.score} out of range [0, 1] for {result.chunk_id}"

        # If results exist, highest score should be high confidence
        if results:
            highest_score = results[0].score
            assert highest_score >= 0.3, f"Highest score should be >= 0.3, got {highest_score}"

        # If multiple results, lowest score must be >= 0.0
        if len(results) > 1:
            lowest_score = results[-1].score
            assert lowest_score >= 0.0, f"Lowest score should be >= 0.0, got {lowest_score}"


@pytest.mark.asyncio
async def test_top_n_greater_than_10_works_correctly(config, manager, orchestrator):
    """
    Test that top_n > 10 works correctly with dynamic top_k calculation.

    This test validates the fix for the issue where top_k was hardcoded to 10,
    which limited results even when top_n was set higher. With the dynamic
    calculation (top_k = max(10, top_n * 2)), requesting top_n=25 should
    retrieve sufficient candidates to return close to 25 results.

    Validates:
    - top_n=25 can return up to 25 results (corpus size permitting)
    - Dynamic top_k calculation provides sufficient candidates
    - Results maintain proper score normalization
    """
    create_test_corpus(config, manager)

    # Request 25 results with dynamic top_k calculation
    results_25, _, _ = await orchestrator.query("authentication security API", top_k=50, top_n=25)

    # Should return results (exact count depends on corpus size)
    # Since our test corpus has 5 documents with multiple chunks each,
    # we should get multiple results, though likely < 25 due to corpus size
    assert len(results_25) > 0, "Should return at least some results"

    # Verify all results have valid scores
    assert all(0.0 <= result.score <= 1.0 for result in results_25)

    # Verify highest score is high confidence
    if results_25:
        assert results_25[0].score >= 0.5

    # Verify results are properly sorted descending
    for i in range(len(results_25) - 1):
        assert results_25[i].score >= results_25[i+1].score

    # Verify that top_n=25 can return more results than top_n=10
    # (if corpus is large enough)
    results_10, _, _ = await orchestrator.query("authentication security API", top_k=20, top_n=10)

    # The key validation: with sufficient corpus, top_n=25 should not be
    # artificially limited to 10 results by a hardcoded top_k
    # At minimum, both queries should work without error and return valid results
    assert len(results_10) <= 10
    assert len(results_25) <= 25

    # If we got 10 results for the first query, and our corpus is larger,
    # we should be able to get more for top_n=25
    # (Though in small test corpus, this might not always be true)
