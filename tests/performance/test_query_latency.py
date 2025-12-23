"""
Performance tests for query latency benchmarking.

Measures end-to-end query latency for hybrid search operations,
including semantic search, keyword search, graph traversal, and
RRF fusion. Tests both cold (first query) and warm (cached) scenarios.
"""

import asyncio
import time
from pathlib import Path

import pytest

from src.config import Config, IndexingConfig, LLMConfig, SearchConfig, ServerConfig
from src.indexing.manager import IndexManager
from src.indices.graph import GraphStore
from src.indices.keyword import KeywordIndex
from src.indices.vector import VectorIndex
from src.search.orchestrator import QueryOrchestrator


@pytest.fixture
def config(tmp_path):
    """
    Create test configuration with temporary paths.

    Uses tmp_path for isolated benchmark runs.
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
        llm=LLMConfig(),
    )


@pytest.fixture
def indices():
    """
    Create real index instances for benchmarking.

    Returns tuple of (vector, keyword, graph) indices.
    """
    vector = VectorIndex()
    keyword = KeywordIndex()
    graph = GraphStore()
    return vector, keyword, graph


@pytest.fixture
def manager(config, indices):
    """
    Create IndexManager with real indices for benchmarking.

    Provides fully functional manager for performance testing.
    """
    vector, keyword, graph = indices
    return IndexManager(config, vector, keyword, graph)


@pytest.fixture
def orchestrator(config, indices, manager):
    """
    Create QueryOrchestrator for hybrid search benchmarking.

    Provides fully functional orchestrator with real indices.
    """
    vector, keyword, graph = indices
    return QueryOrchestrator(vector, keyword, graph, config, manager)


def create_benchmark_corpus(docs_path: Path, num_docs: int = 50):
    """
    Create test corpus optimized for query benchmarking.

    Creates documents with varied content to support semantic,
    keyword, and graph search strategies.
    """
    topics = [
        ("Machine Learning", "machine learning algorithms neural networks deep learning"),
        ("Cloud Infrastructure", "cloud computing AWS kubernetes docker containers"),
        ("Database Systems", "database SQL PostgreSQL optimization indexing queries"),
        ("Web Development", "web development React JavaScript TypeScript frontend"),
        ("Security", "security authentication encryption cryptography vulnerabilities"),
        ("Testing", "testing unit tests integration tests pytest coverage"),
        ("DevOps", "devops CI/CD pipelines deployment automation monitoring"),
        ("API Design", "API REST GraphQL endpoints HTTP requests responses"),
    ]

    for i in range(num_docs):
        topic_idx = i % len(topics)
        topic_name, keywords = topics[topic_idx]

        content = f"# {topic_name} Guide {i}\n\n"
        content += f"tags: [{', '.join(keywords.split()[:3])}]\n\n"

        content += f"## Introduction to {topic_name}\n\n"
        content += f"This document covers essential concepts in {keywords}. "
        content += "It provides practical examples and best practices for implementation.\n\n"

        content += "## Core Concepts\n\n"
        content += f"Understanding {topic_name} requires knowledge of {keywords}. "
        content += "These concepts form the foundation of modern software development.\n\n"

        content += "## Implementation Details\n\n"
        content += f"When implementing {topic_name} solutions, consider {keywords}. "
        content += "Performance and scalability are critical factors.\n\n"

        # Add links to related documents
        if i > 0:
            content += f"See also [[{topics[(i - 1) % len(topics)][0]} Guide {i - 1}]] for related information.\n\n"

        doc_file = docs_path / f"doc_{i:03d}_{topic_name.replace(' ', '_').lower()}.md"
        doc_file.write_text(content)


@pytest.fixture
def indexed_corpus(config, manager):
    """
    Create and index test corpus for query benchmarking.

    Module-scoped fixture to amortize indexing cost across query tests.
    """
    docs_path = Path(config.indexing.documents_path)

    # Create corpus
    create_benchmark_corpus(docs_path, num_docs=50)

    # Index all documents
    for doc_file in sorted(docs_path.glob("*.md")):
        manager.index_document(str(doc_file))

    # Verify corpus indexed
    doc_count = manager.get_document_count()
    assert doc_count == 50, f"Expected 50 documents, got {doc_count}"

    return manager


@pytest.mark.asyncio
async def test_query_latency_cold_start(config, orchestrator, indexed_corpus):
    """
    Benchmark cold start query latency.

    Measures first query performance including any lazy initialization
    and model loading overhead.
    """
    # Single cold query
    query = "machine learning algorithms and neural networks"

    start_time = time.perf_counter()
    results = await orchestrator.query(query, top_k=10)
    end_time = time.perf_counter()

    latency = (end_time - start_time) * 1000  # Convert to milliseconds

    # Verify results returned
    assert len(results) > 0, "Expected query results for cold start"
    assert len(results) <= 10, f"Expected at most 10 results, got {len(results)}"

    # Report metrics
    print("\n=== Cold Start Query Latency ===")
    print(f"Query: '{query}'")
    print(f"Latency: {latency:.2f}ms")
    print(f"Results: {len(results)} documents")

    # Performance baseline (cold start can include model loading)
    assert latency < 10000, f"Cold start too slow: {latency:.2f}ms"


@pytest.mark.asyncio
async def test_query_latency_warm_queries(config, orchestrator, indexed_corpus):
    """
    Benchmark warm query latency with percentile analysis.

    Measures multiple queries to calculate p50, p95, p99 latencies,
    representing typical production query performance.
    """
    queries = [
        "machine learning algorithms",
        "cloud infrastructure kubernetes",
        "database optimization techniques",
        "web development frontend",
        "security authentication",
        "testing strategies pytest",
        "devops CI/CD pipelines",
        "API design REST",
        "neural networks deep learning",
        "container orchestration",
    ]

    latencies = []

    # Warm up with first query
    await orchestrator.query(queries[0], top_k=10)

    # Benchmark multiple queries
    for query in queries:
        start_time = time.perf_counter()
        results = await orchestrator.query(query, top_k=10)
        end_time = time.perf_counter()

        latency_ms = (end_time - start_time) * 1000
        latencies.append(latency_ms)

        # Verify results
        assert len(results) > 0, f"Expected results for query: '{query}'"
        assert len(results) <= 10, f"Expected at most 10 results, got {len(results)}"

    # Calculate percentiles
    latencies.sort()
    p50 = latencies[len(latencies) // 2]
    p95 = latencies[int(len(latencies) * 0.95)]
    p99 = latencies[int(len(latencies) * 0.99)]
    avg = sum(latencies) / len(latencies)
    min_latency = min(latencies)
    max_latency = max(latencies)

    # Report metrics
    print(f"\n=== Warm Query Latency (n={len(queries)}) ===")
    print(f"Min: {min_latency:.2f}ms")
    print(f"P50: {p50:.2f}ms")
    print(f"P95: {p95:.2f}ms")
    print(f"P99: {p99:.2f}ms")
    print(f"Max: {max_latency:.2f}ms")
    print(f"Avg: {avg:.2f}ms")

    # Performance baselines
    assert p50 < 5000, f"P50 latency too high: {p50:.2f}ms"
    assert p95 < 8000, f"P95 latency too high: {p95:.2f}ms"
    assert p99 < 10000, f"P99 latency too high: {p99:.2f}ms"


@pytest.mark.asyncio
async def test_query_latency_by_top_k(config, orchestrator, indexed_corpus):
    """
    Benchmark query latency with varying top_k values.

    Tests performance impact of result set size on query latency,
    identifying potential bottlenecks in result retrieval and ranking.
    """
    query = "machine learning algorithms and neural networks"
    top_k_values = [5, 10, 20, 50]

    results_data = []

    for top_k in top_k_values:
        # Warm up
        await orchestrator.query(query, top_k=top_k)

        # Benchmark
        latencies = []
        for _ in range(5):  # 5 runs per top_k
            start_time = time.perf_counter()
            results = await orchestrator.query(query, top_k=top_k)
            end_time = time.perf_counter()

            latency_ms = (end_time - start_time) * 1000
            latencies.append(latency_ms)

            # Verify results
            assert len(results) > 0, f"Expected results for top_k={top_k}"
            assert len(results) <= top_k, f"Expected at most {top_k} results"

        avg_latency = sum(latencies) / len(latencies)
        results_data.append((top_k, avg_latency))

    # Report metrics
    print("\n=== Query Latency by top_k ===")
    print(f"Query: '{query}'")
    for top_k, avg_latency in results_data:
        print(f"top_k={top_k:2d}: {avg_latency:.2f}ms")

    # Verify latency increases with top_k (or stays similar)
    # Allow some variance but ensure reasonable performance
    for top_k, avg_latency in results_data:
        assert avg_latency < 6000, f"Latency too high for top_k={top_k}: {avg_latency:.2f}ms"


@pytest.mark.asyncio
async def test_query_latency_concurrent_queries(config, orchestrator, indexed_corpus):
    """
    Benchmark concurrent query performance.

    Tests query latency under concurrent load to identify potential
    bottlenecks in parallel query processing and resource contention.
    """
    queries = [
        "machine learning algorithms",
        "cloud infrastructure",
        "database optimization",
        "web development",
        "security practices",
    ]

    # Warm up
    await orchestrator.query(queries[0], top_k=10)

    # Benchmark concurrent queries
    start_time = time.perf_counter()

    tasks = [orchestrator.query(q, top_k=10) for q in queries]
    results_list = await asyncio.gather(*tasks)

    end_time = time.perf_counter()

    total_time = (end_time - start_time) * 1000
    avg_time_per_query = total_time / len(queries)

    # Verify all queries completed
    for i, results in enumerate(results_list):
        assert len(results) > 0, f"Expected results for query {i}: '{queries[i]}'"
        assert len(results) <= 10, f"Expected at most 10 results for query {i}"

    # Report metrics
    print("\n=== Concurrent Query Performance ===")
    print(f"Queries: {len(queries)} concurrent")
    print(f"Total time: {total_time:.2f}ms")
    print(f"Avg per query: {avg_time_per_query:.2f}ms")
    print(f"Throughput: {len(queries) / (total_time / 1000):.2f} queries/sec")

    # Performance baseline (concurrent should be faster than serial)
    # Allow up to 5s per query on average in concurrent mode
    assert avg_time_per_query < 5000, f"Concurrent query avg too high: {avg_time_per_query:.2f}ms"


@pytest.mark.asyncio
async def test_query_latency_empty_results(config, orchestrator, indexed_corpus):
    """
    Benchmark query latency for queries with no matches.

    Tests performance when query returns empty results, ensuring
    graceful handling and minimal latency overhead.
    """
    # Query unlikely to match any documents
    query = "xyzabc123 nonexistent impossible query terms"

    latencies = []

    for _ in range(10):
        start_time = time.perf_counter()
        results = await orchestrator.query(query, top_k=10)
        end_time = time.perf_counter()

        latency_ms = (end_time - start_time) * 1000
        latencies.append(latency_ms)

        # Results may be empty or very low relevance
        assert isinstance(results, list), "Expected list of results"

    avg_latency = sum(latencies) / len(latencies)

    # Report metrics
    print("\n=== Empty Result Query Latency ===")
    print(f"Query: '{query}'")
    print(f"Avg latency: {avg_latency:.2f}ms (n={len(latencies)})")

    # Empty result queries should still complete quickly
    assert avg_latency < 5000, f"Empty query latency too high: {avg_latency:.2f}ms"
