"""Search quality evaluation tests using searchkernel.eval harness.

Uses the golden-set regression testing framework to measure real search-quality
metrics (recall@k, nDCG@k, MRR, AP) against a small fixture document corpus.
Ensures future chunking/ranking changes get a quality regression signal.
"""

from pathlib import Path

import pytest
from searchkernel.eval.golden import GoldenEntry, GoldenSet
from searchkernel.eval.runner import run_eval
from searchkernel.indices.graph import GraphStore
from searchkernel.indices.keyword import KeywordIndex
from searchkernel.indices.vector import VectorIndex
from searchkernel.search.orchestrator import SearchOrchestrator

from mcp_markdown_ragdocs.config import Config, IndexingConfig, LLMConfig, SearchConfig
from mcp_markdown_ragdocs.indexing.manager import IndexManager
from tests.conftest import create_test_document


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def search_quality_config(tmp_path):
    """Config for search quality tests with isolated indices."""
    docs_path = tmp_path / "docs"
    docs_path.mkdir()
    return Config(
        indexing=IndexingConfig(
            documents_path=str(docs_path), index_path=str(tmp_path / "indices")
        ),
        search=SearchConfig(
            semantic_weight=1.0,
            keyword_weight=1.0,
            recency_bias=0.0,  # Disable recency for reproducible test results
        ),
        llm=LLMConfig(embedding_model="BAAI/bge-small-en-v1.5"),
    )


# ============================================================================
# Fixture Corpus
# ============================================================================


def _create_fixture_corpus(docs_path: Path) -> None:
    """Create a small fixture corpus covering distinct documentation topics."""
    create_test_document(
        docs_path,
        "authentication",
        """# Authentication and Security

Authentication is the process of verifying user identity and credentials.
Implement OAuth2 for secure user login flows.
Use multi-factor authentication to protect sensitive accounts.
Password hashing with bcrypt or Argon2 is essential.
""",
    )

    create_test_document(
        docs_path,
        "database_migrations",
        """# Database Schema Migrations

Migrations manage schema changes across environments.
Use version control for all migration files.
Always test migrations on staging before production.
Rollback procedures must be defined for each migration.
""",
    )

    create_test_document(
        docs_path,
        "logging_configuration",
        """# Logging and Monitoring Setup

Structured logging helps debug production issues.
Configure log levels (DEBUG, INFO, WARN, ERROR) appropriately.
Use centralized log aggregation for distributed systems.
Set up alerts for critical errors and performance anomalies.
""",
    )

    create_test_document(
        docs_path,
        "deployment_process",
        """# Deployment and Release Management

Continuous integration pipelines automate testing and deployment.
Use semantic versioning for release tags.
Blue-green deployments minimize downtime.
Health checks must pass before marking deployment successful.
""",
    )

    create_test_document(
        docs_path,
        "api_design",
        """# REST API Design Principles

API endpoints should follow consistent naming conventions.
Use HTTP status codes correctly (200, 201, 400, 404, 500).
Implement pagination for large result sets.
API documentation must be kept in sync with implementation.
""",
    )

    create_test_document(
        docs_path,
        "testing_strategy",
        """# Testing Framework and Best Practices

Write unit tests for business logic and edge cases.
Integration tests verify component interactions.
End-to-end tests validate complete user workflows.
Aim for at least 80% code coverage.
""",
    )

    create_test_document(
        docs_path,
        "caching_strategies",
        """# Caching and Performance Optimization

Cache frequently accessed data to reduce database load.
Use TTL (time-to-live) for cache entries.
Implement cache invalidation strategies.
Consider Redis or Memcached for distributed caching.
""",
    )


# ============================================================================
# Golden Set for Search Quality Evaluation
# ============================================================================


def _create_golden_set() -> GoldenSet:
    """Create a golden set with known-relevant document IDs for each query.

    Each query is designed to test specific retrieval aspects:
    - Exact term matches (testing keyword matching)
    - Semantic synonyms (testing semantic matching)
    - Multi-word intent (testing phrase understanding)
    """
    entries = [
        # Query 1: Exact keyword match
        GoldenEntry(
            query="authentication and security",
            relevant_ids=["authentication"],
        ),
        # Query 2: Semantic synonym (login = authentication)
        GoldenEntry(
            query="how to implement user login",
            relevant_ids=["authentication"],
        ),
        # Query 3: Testing-related query
        GoldenEntry(
            query="unit tests and integration tests",
            relevant_ids=["testing_strategy"],
        ),
        # Query 4: Deployment workflow
        GoldenEntry(
            query="continuous integration deployment pipeline",
            relevant_ids=["deployment_process"],
        ),
        # Query 5: Multi-document query (relevant to multiple docs)
        GoldenEntry(
            query="database management and schema",
            relevant_ids=["database_migrations"],
        ),
        # Query 6: Caching and optimization
        GoldenEntry(
            query="cache and performance optimization",
            relevant_ids=["caching_strategies"],
        ),
        # Query 7: API development
        GoldenEntry(
            query="REST endpoints HTTP status codes",
            relevant_ids=["api_design"],
        ),
        # Query 8: Monitoring and operations
        GoldenEntry(
            query="logs aggregation and error alerts",
            relevant_ids=["logging_configuration"],
        ),
    ]
    return GoldenSet(entries=entries)


# ============================================================================
# Search Quality Evaluation Test
# ============================================================================


@pytest.mark.asyncio
@pytest.mark.real_embeddings
@pytest.mark.slow
async def test_search_quality_regression(search_quality_config, shared_embedding_model):
    """Test search quality metrics against a golden-set fixture corpus.

    Measures recall@k, nDCG@k, MRR, and AP using the searchkernel.eval harness.
    Ensures that changes to chunking, ranking, or fusion logic maintain or
    improve search quality.

    Asserts that mean_recall_at_k >= 0.7 on a carefully designed fixture set.
    Also verifies that all metrics (nDCG@k, MRR, AP) are computed.
    """
    # ========================================================================
    # 1. Build fixture corpus and indices
    # ========================================================================
    docs_path = Path(search_quality_config.indexing.documents_path)

    # Create fixture documents
    _create_fixture_corpus(docs_path)

    # Create indices
    vector = VectorIndex(embedding_model=shared_embedding_model)
    keyword = KeywordIndex()
    graph = GraphStore()

    # Create manager and orchestrator
    manager = IndexManager(search_quality_config, vector, keyword, graph)
    orchestrator = SearchOrchestrator(
        vector, keyword, graph, search_quality_config, manager
    )

    # Index all fixture documents
    for doc_file in sorted(docs_path.glob("*.md")):
        manager.index_document(str(doc_file))

    # Build concept vocabulary for semantic expansion
    vector.build_concept_vocabulary()

    # ========================================================================
    # 2. Define search function for eval harness
    # ========================================================================

    async def search_async(query: str) -> list[str]:
        """Run orchestrator query and return ordered list of document IDs."""
        results, _stats, _ = await orchestrator.query(
            query, top_k=10, top_n=5
        )
        # Use record_id (document ID), not chunk_id, for golden-set matching
        return [r.record_id for r in results]

    # Synchronous wrapper for eval harness (which expects Callable[[str], list[str]])
    import asyncio

    def search_fn(query: str) -> list[str]:
        """Synchronous wrapper around async orchestrator.query."""
        # This is called from run_eval (sync context) while inside an async test.
        # Use nest_asyncio to allow running async code from sync wrapper.
        try:
            import nest_asyncio
            nest_asyncio.apply()
        except ImportError:
            pass

        # Try to use existing event loop if available, else create new one
        try:
            loop = asyncio.get_running_loop()
            # Already in async context, can't use run_until_complete
            # Fall back to creating task in current loop (won't work from sync)
        except RuntimeError:
            # No running loop, safe to create new one
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(search_async(query))
            finally:
                loop.close()

        # If we get here, we're in an async context being called from sync.
        # This requires nest_asyncio to work properly.
        return asyncio.run(search_async(query))

    # ========================================================================
    # 3. Run evaluation against golden set
    # ========================================================================
    golden_set = _create_golden_set()
    report = run_eval(golden_set, search_fn, k=5)

    # ========================================================================
    # 4. Assert quality metrics
    # ========================================================================

    # Mean recall@5 should be at least 0.7 (70% of relevant docs found in top 5)
    assert (
        report.mean_recall_at_k is not None
    ), "mean_recall_at_k should be computed"
    assert (
        report.mean_recall_at_k >= 0.7
    ), f"mean_recall@5={report.mean_recall_at_k:.2f}, expected >= 0.7"

    # Verify other metrics were computed
    assert report.mean_ndcg_at_k is not None, "mean_ndcg_at_k should be computed"
    assert report.mean_mrr is not None, "mean_mrr should be computed"
    assert report.mean_ap is not None, "mean_ap should be computed"

    # Log metrics for observability
    import logging
    log = logging.getLogger(__name__)
    log.info(
        f"Search Quality Metrics (k={report.k}, n={len(golden_set.entries)} queries)"
    )
    log.info(f"  Mean Recall@{report.k}: {report.mean_recall_at_k:.3f}")
    log.info(f"  Mean nDCG@{report.k}: {report.mean_ndcg_at_k:.3f}")
    log.info(f"  Mean MRR: {report.mean_mrr:.3f}")
    log.info(f"  Mean AP: {report.mean_ap:.3f}")
    if report.latency_p50_ms:
        log.info(f"  Latency p50: {report.latency_p50_ms:.1f}ms")
    if report.latency_p95_ms:
        log.info(f"  Latency p95: {report.latency_p95_ms:.1f}ms")
    if report.latency_p99_ms:
        log.info(f"  Latency p99: {report.latency_p99_ms:.1f}ms")
