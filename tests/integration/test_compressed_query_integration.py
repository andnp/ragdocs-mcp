"""
Integration tests for query_documents MCP tool compression features.

Tests cover:
- Full compression pipeline through MCP server handler
- Score threshold filtering
- Semantic deduplication
- Response format with compression stats
- Parameter handling (top_n, min_score, similarity_threshold)

These tests use real embeddings and indices, avoiding mocks.
"""

from pathlib import Path
from typing import Generator

import numpy as np
import pytest

from src.compression.thresholding import filter_by_score
from src.config import Config, IndexingConfig, LLMConfig, SearchConfig, ServerConfig, ChunkingConfig
from src.indexing.manager import IndexManager
from src.indices.graph import GraphStore
from src.indices.keyword import KeywordIndex
from src.indices.vector import VectorIndex
from src.models import CompressionStats
from src.search.orchestrator import SearchOrchestrator
from src.search.pipeline import SearchPipelineConfig


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture(scope="module")
def integration_config(tmp_path_factory) -> Config:
    """
    Create module-scoped configuration for integration tests.

    Sets up paths and configuration needed for full pipeline testing.
    """
    base_path = tmp_path_factory.mktemp("compressed_query_test")
    docs_path = base_path / "documents"
    index_path = base_path / "indices"
    docs_path.mkdir(parents=True, exist_ok=True)
    index_path.mkdir(parents=True, exist_ok=True)

    return Config(
        server=ServerConfig(),
        indexing=IndexingConfig(
            documents_path=str(docs_path),
            index_path=str(index_path),
            recursive=True,
        ),
        parsers={"**/*.md": "MarkdownParser"},
        search=SearchConfig(
            semantic_weight=1.0,
            keyword_weight=1.0,
            recency_bias=0.5,
            rrf_k_constant=60,
        ),
        llm=LLMConfig(embedding_model="BAAI/bge-small-en-v1.5"),
        document_chunking=ChunkingConfig(),
        memory_chunking=ChunkingConfig(),
    )


@pytest.fixture(scope="module")
def embedding_model(shared_embedding_model):
    """
    Reuse session-scoped embedding model for efficiency.

    Avoids loading the embedding model multiple times during tests.
    """
    return shared_embedding_model


@pytest.fixture(scope="module")
def integration_indices(
    embedding_model,
) -> Generator[tuple[VectorIndex, KeywordIndex, GraphStore], None, None]:
    """
    Create module-scoped indices for integration tests.

    These indices are shared across tests in this module.
    """
    vector = VectorIndex(embedding_model=embedding_model)
    keyword = KeywordIndex()
    graph = GraphStore()
    yield vector, keyword, graph


@pytest.fixture(scope="module")
def integration_manager(
    integration_config: Config,
    integration_indices: tuple[VectorIndex, KeywordIndex, GraphStore],
) -> IndexManager:
    """
    Create module-scoped IndexManager for integration tests.

    Provides access to index management functionality.
    """
    vector, keyword, graph = integration_indices
    return IndexManager(integration_config, vector, keyword, graph)


@pytest.fixture(scope="module")
def integration_orchestrator(
    integration_indices: tuple[VectorIndex, KeywordIndex, GraphStore],
    integration_config: Config,
    integration_manager: IndexManager,
) -> SearchOrchestrator:
    """
    Create module-scoped SearchOrchestrator for integration tests.

    Provides query execution capabilities.
    """
    vector, keyword, graph = integration_indices
    return SearchOrchestrator(vector, keyword, graph, integration_config, integration_manager)


@pytest.fixture(scope="module")
def indexed_documents(
    integration_config: Config,
    integration_manager: IndexManager,
) -> list[str]:
    """
    Create and index test documents for compression testing.

    Returns list of file paths for indexed documents.
    """
    docs_path = Path(integration_config.indexing.documents_path)

    # Create documents with intentional similarity patterns
    documents = {
        "python_basics.md": """# Python Basics

## Introduction to Python
Python is a high-level programming language known for readability.
It supports multiple programming paradigms.

## Variables and Types
Python uses dynamic typing. Variables don't need explicit type declarations.
Common types include int, float, str, list, dict.

## Control Flow
Python uses indentation for code blocks.
If statements and loops are fundamental control structures.
""",
        "python_intro.md": """# Introduction to Python Programming

## What is Python?
Python is a high-level, interpreted programming language with clear syntax.
It emphasizes code readability and supports multiple paradigms.

## Getting Started
Install Python from python.org.
Use pip for package management.

## Basic Syntax
Python uses whitespace indentation.
Comments start with # symbol.
""",
        "rust_basics.md": """# Rust Programming

## Introduction to Rust
Rust is a systems programming language focused on safety and performance.
It prevents memory errors at compile time.

## Ownership System
Rust's ownership model ensures memory safety without garbage collection.
Each value has a single owner.

## Concurrency
Rust provides fearless concurrency through its type system.
Data races are prevented at compile time.
""",
        "javascript_intro.md": """# JavaScript Basics

## What is JavaScript?
JavaScript is a dynamic scripting language for web development.
It runs in browsers and Node.js environments.

## Variables
Use let and const for variable declarations.
Avoid var in modern JavaScript.

## Functions
JavaScript supports first-class functions.
Arrow functions provide concise syntax.
""",
        "database_design.md": """# Database Design

## Relational Databases
SQL databases use tables with rows and columns.
Relationships are established through foreign keys.

## NoSQL Databases
Document stores like MongoDB use JSON-like documents.
Key-value stores provide simple, fast lookups.

## Query Optimization
Indexing improves query performance.
Query plans help analyze execution strategies.
""",
    }

    file_paths = []
    for filename, content in documents.items():
        file_path = docs_path / filename
        file_path.write_text(content)
        file_paths.append(str(file_path))
        integration_manager.index_document(str(file_path))

    return file_paths


# ============================================================================
# Compression Pipeline Tests
# ============================================================================


class TestCompressionPipeline:
    """Tests for the full compression pipeline."""

    @pytest.mark.asyncio
    async def test_query_with_compression_returns_results(
        self,
        integration_orchestrator: SearchOrchestrator,
        indexed_documents: list[str],
    ) -> None:
        """
        Tests that compressed query returns valid results with stats.

        Verifies the basic flow: query → filter → deduplicate → return.
        """
        # Execute query
        results, _, _ = await integration_orchestrator.query(
            "What is Python programming?",
            top_k=20,
            top_n=20,
        )

        assert len(results) > 0

        # Apply compression
        filtered = filter_by_score(results, min_score=0.3)
        assert len(filtered) <= len(results)

    @pytest.mark.asyncio
    async def test_score_threshold_filters_results(
        self,
        integration_orchestrator: SearchOrchestrator,
        indexed_documents: list[str],
    ) -> None:
        """
        Tests that score threshold effectively filters low-relevance results.

        Higher threshold should result in fewer results.
        """
        results, _, _ = await integration_orchestrator.query(
            "Python variables and types",
            top_k=20,
            top_n=20,
        )

        # Low threshold: most results pass
        filtered_low = filter_by_score(results, min_score=0.1)

        # High threshold: fewer results pass
        filtered_high = filter_by_score(results, min_score=0.5)

        assert len(filtered_high) <= len(filtered_low)

    @pytest.mark.asyncio
    async def test_deduplication_reduces_similar_results(
        self,
        integration_orchestrator: SearchOrchestrator,
        indexed_documents: list[str],
        embedding_model,
    ) -> None:
        """
        Tests that deduplication merges semantically similar chunks.

        Query about Python should return similar chunks from python_basics.md
        and python_intro.md that get deduplicated.
        """
        pipeline_config = SearchPipelineConfig(
            min_confidence=0.0,
            dedup_enabled=True,
            dedup_threshold=0.85,
        )

        results, stats, _ = await integration_orchestrator.query(
            "Introduction to Python programming language",
            top_k=20,
            top_n=20,
            pipeline_config=pipeline_config,
        )

        assert stats.original_count >= 0
        assert stats.after_dedup <= stats.original_count


# ============================================================================
# Response Format Tests
# ============================================================================


class TestCompressionStats:
    """Tests for compression statistics tracking."""

    @pytest.mark.asyncio
    async def test_compression_stats_structure(
        self,
        integration_orchestrator: SearchOrchestrator,
        indexed_documents: list[str],
        embedding_model,
    ) -> None:
        """
        Tests that compression stats are correctly computed.

        Verifies all stat fields are populated with valid values.
        """
        pipeline_config = SearchPipelineConfig(
            min_confidence=0.3,
            dedup_enabled=True,
            dedup_threshold=0.85,
        )

        results, stats, _ = await integration_orchestrator.query(
            "database query optimization",
            top_k=20,
            top_n=20,
            pipeline_config=pipeline_config,
        )

        assert stats.original_count >= stats.after_threshold
        assert stats.after_threshold >= stats.after_dedup
        assert stats.clusters_merged >= 0

    def test_compression_stats_to_dict(self) -> None:
        """
        Tests CompressionStats.to_dict() serialization.

        Verifies all fields are present in dictionary output.
        """
        stats = CompressionStats(
            original_count=20,
            after_threshold=15,
            after_content_dedup=14,
            after_ngram_dedup=13,
            after_dedup=12,
            after_doc_limit=10,
            clusters_merged=5,
        )

        stats_dict = stats.to_dict()

        assert stats_dict["original_count"] == 20
        assert stats_dict["after_threshold"] == 15
        assert stats_dict["after_content_dedup"] == 14
        assert stats_dict["after_dedup"] == 12
        assert stats_dict["after_doc_limit"] == 10
        assert stats_dict["clusters_merged"] == 5


# ============================================================================
# Parameter Handling Tests
# ============================================================================


class TestParameterHandling:
    """Tests for compression parameter handling."""

    @pytest.mark.asyncio
    async def test_top_n_limits_final_results(
        self,
        integration_orchestrator: SearchOrchestrator,
        indexed_documents: list[str],
        embedding_model,
    ) -> None:
        """
        Tests that top_n parameter limits final result count.

        After all compression, results should not exceed top_n.
        """
        pipeline_config = SearchPipelineConfig(
            min_confidence=0.1,
            dedup_enabled=True,
            dedup_threshold=0.85,
        )

        results, _, _ = await integration_orchestrator.query(
            "programming language introduction",
            top_k=20,
            top_n=3,
            pipeline_config=pipeline_config,
        )

        assert len(results) <= 3

    @pytest.mark.asyncio
    async def test_min_score_parameter_effect(
        self,
        integration_orchestrator: SearchOrchestrator,
        indexed_documents: list[str],
    ) -> None:
        """
        Tests that min_score parameter controls threshold filtering.

        Different min_score values should produce different result counts.
        """
        results, _, _ = await integration_orchestrator.query(
            "Rust ownership memory safety",
            top_k=20,
            top_n=20,
        )

        filtered_0_1 = filter_by_score(results, min_score=0.1)
        filtered_0_5 = filter_by_score(results, min_score=0.5)
        filtered_0_9 = filter_by_score(results, min_score=0.9)

        # Higher threshold = fewer or equal results
        assert len(filtered_0_9) <= len(filtered_0_5) <= len(filtered_0_1)

    @pytest.mark.asyncio
    async def test_similarity_threshold_parameter_effect(
        self,
        integration_orchestrator: SearchOrchestrator,
        indexed_documents: list[str],
        embedding_model,
    ) -> None:
        """
        Tests that similarity_threshold parameter controls deduplication.

        Lower threshold should merge more results.
        """
        config_high = SearchPipelineConfig(
            min_confidence=0.2,
            dedup_enabled=True,
            dedup_threshold=0.95,
        )

        config_low = SearchPipelineConfig(
            min_confidence=0.2,
            dedup_enabled=True,
            dedup_threshold=0.7,
        )

        _, stats_high, _ = await integration_orchestrator.query(
            "Python programming basics",
            top_k=20,
            top_n=20,
            pipeline_config=config_high,
        )

        _, stats_low, _ = await integration_orchestrator.query(
            "Python programming basics",
            top_k=20,
            top_n=20,
            pipeline_config=config_low,
        )

        assert stats_low.after_dedup <= stats_high.after_dedup


# ============================================================================
# Edge Case Tests
# ============================================================================


class TestEdgeCases:
    """Tests for edge cases in compression pipeline."""

    @pytest.mark.asyncio
    async def test_query_with_no_results(
        self,
        integration_orchestrator: SearchOrchestrator,
        indexed_documents: list[str],
    ) -> None:
        """
        Tests compression with query that returns no results.

        Should handle empty results gracefully.
        """
        results, _, _ = await integration_orchestrator.query(
            "xyzzy completely unrelated nonsense query 12345",
            top_k=5,
            top_n=5,
        )

        filtered = filter_by_score(results, min_score=0.9)

        assert isinstance(filtered, list)
        # May be empty if no relevant results

    @pytest.mark.asyncio
    async def test_single_result_passthrough(
        self,
        integration_orchestrator: SearchOrchestrator,
        indexed_documents: list[str],
        embedding_model,
    ) -> None:
        """
        Tests that single result passes through unchanged.

        Deduplication with one item should be a no-op.
        """
        pipeline_config = SearchPipelineConfig(
            min_confidence=0.1,
            dedup_enabled=True,
            dedup_threshold=0.85,
        )

        results, stats, _ = await integration_orchestrator.query(
            "Rust fearless concurrency type system",
            top_k=5,
            top_n=1,
            pipeline_config=pipeline_config,
        )

        assert len(results) <= 1
        if len(results) == 1:
            assert stats.clusters_merged == 0

    @pytest.mark.asyncio
    async def test_all_results_filtered_by_threshold(
        self,
        integration_orchestrator: SearchOrchestrator,
        indexed_documents: list[str],
    ) -> None:
        """
        Tests behavior when all results are below threshold.

        Very high threshold may filter everything out.
        """
        results, _, _ = await integration_orchestrator.query(
            "general programming concepts",
            top_k=10,
            top_n=10,
        )

        # Very high threshold
        filtered = filter_by_score(results, min_score=0.99)

        assert isinstance(filtered, list)
        # Likely empty with such high threshold


# ============================================================================
# Integration with Embeddings Tests
# ============================================================================


class TestEmbeddingsIntegration:
    @pytest.mark.asyncio
    async def test_embedding_model_generates_vectors(
        self,
        integration_orchestrator: SearchOrchestrator,
        indexed_documents: list[str],
        embedding_model,
    ) -> None:
        results, _, _ = await integration_orchestrator.query(
            "Python introduction",
            top_k=5,
            top_n=5,
        )

        if len(results) > 0:
            embeddings = [
                embedding_model.get_text_embedding(r.content)
                for r in results
            ]

            assert len(embeddings) == len(results)
            assert all(len(emb) == 384 for emb in embeddings)

    @pytest.mark.asyncio
    async def test_embeddings_enable_similarity_detection(
        self,
        integration_orchestrator: SearchOrchestrator,
        indexed_documents: list[str],
        embedding_model,
    ) -> None:
        results, _, _ = await integration_orchestrator.query(
            "Python programming language overview",
            top_k=10,
            top_n=10,
        )

        if len(results) > 1:
            embeddings = np.array([
                embedding_model.get_text_embedding(r.content)
                for r in results
            ])

            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1, norms)
            normalized = embeddings / norms
            sim_matrix = np.dot(normalized, normalized.T)

            diagonal = np.diag(sim_matrix)
            np.testing.assert_array_almost_equal(diagonal, np.ones(len(results)))

            np.testing.assert_array_almost_equal(sim_matrix, sim_matrix.T)
