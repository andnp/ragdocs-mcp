from pathlib import Path

import pytest

from src.config import Config, IndexingConfig, LLMConfig, SearchConfig, ServerConfig
from src.indexing.manager import IndexManager
from src.indices.graph import GraphStore
from src.indices.keyword import KeywordIndex
from src.indices.vector import VectorIndex
from src.models import ChunkResult
from src.search.orchestrator import SearchOrchestrator
from tests.conftest import create_test_document


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def config_query_expansion(tmp_path):
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
            recency_bias=0.0,  # Disable recency for predictable tests
            rrf_k_constant=60,
            rerank_enabled=False,
            dedup_enabled=False,
        ),
        llm=LLMConfig(embedding_model="BAAI/bge-small-en-v1.5"),
    )


@pytest.fixture
def config_reranking(tmp_path):
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
            recency_bias=0.0,
            rrf_k_constant=60,
            rerank_enabled=True,
            rerank_model="cross-encoder/ms-marco-MiniLM-L-6-v2",
            rerank_top_n=5,
            dedup_enabled=False,
        ),
        llm=LLMConfig(embedding_model="BAAI/bge-small-en-v1.5"),
    )


@pytest.fixture
def indices(shared_embedding_model):
    vector = VectorIndex(embedding_model=shared_embedding_model)
    keyword = KeywordIndex()
    graph = GraphStore()
    return vector, keyword, graph


# ============================================================================
# Query Expansion Integration Tests
# ============================================================================


class TestQueryExpansionInOrchestrator:
    """Integration tests for query expansion in search pipeline."""

    @pytest.mark.asyncio
    async def test_query_expansion_in_orchestrator_basic(
        self,
        config_query_expansion,
        indices,
    ):
        """
        Expanded query used in search returns results.

        Verifies that query expansion integrates with orchestrator
        without errors.
        """
        vector, keyword, graph = indices
        manager = IndexManager(config_query_expansion, vector, keyword, graph)
        orchestrator = SearchOrchestrator(
            vector, keyword, graph, config_query_expansion, manager
        )

        docs_path = Path(config_query_expansion.indexing.documents_path)

        # Create documents with related terms
        create_test_document(
            docs_path,
            "authentication",
            """# Authentication Guide

Authentication is the process of verifying user identity.
Users must provide valid credentials to authenticate.
The authentication system supports multiple auth providers.
""",
        )
        create_test_document(
            docs_path,
            "authorization",
            """# Authorization Overview

Authorization determines what authenticated users can access.
Permissions are granted based on user roles.
""",
        )

        # Index documents
        for doc_file in docs_path.glob("*.md"):
            manager.index_document(str(doc_file))

        # Build vocabulary for expansion
        vector.build_concept_vocabulary()

        # Query should work with expansion enabled
        results, stats = await orchestrator.query("auth", top_k=10, top_n=5)

        # Should find results
        assert len(results) > 0

        # Should be ChunkResult objects
        assert all(isinstance(r, ChunkResult) for r in results)

    @pytest.mark.asyncio
    async def test_query_expansion_finds_related_documents(
        self,
        config_query_expansion,
        indices,
    ):
        """
        Query expansion helps find documents with related but different terms.

        Short query "auth" should find documents containing "authentication"
        through vocabulary expansion.
        """
        vector, keyword, graph = indices
        manager = IndexManager(config_query_expansion, vector, keyword, graph)
        orchestrator = SearchOrchestrator(
            vector, keyword, graph, config_query_expansion, manager
        )

        docs_path = Path(config_query_expansion.indexing.documents_path)

        # Document uses "authentication", not "auth"
        create_test_document(
            docs_path,
            "security",
            """# Security Documentation

The authentication system verifies user credentials.
Strong authentication is required for all users.
Multi-factor authentication is recommended.
""",
        )
        create_test_document(
            docs_path,
            "unrelated",
            """# Database Configuration

PostgreSQL configuration for production environments.
Connection pooling and query optimization.
""",
        )

        for doc_file in docs_path.glob("*.md"):
            manager.index_document(str(doc_file))

        vector.build_concept_vocabulary()

        # Query with short form
        results, _ = await orchestrator.query("auth", top_k=10, top_n=5)

        # Should find security doc (contains "authentication")
        result_doc_ids = [r.doc_id for r in results]
        assert "security" in result_doc_ids or any("security" in did for did in result_doc_ids)


# ============================================================================
# Re-ranking Integration Tests
# ============================================================================


class TestRerankingInPipeline:
    """Integration tests for re-ranking in search pipeline."""

    @pytest.mark.asyncio
    @pytest.mark.slow  # Re-ranking loads model, may be slow
    async def test_reranking_in_pipeline_basic(
        self,
        config_reranking,
        shared_embedding_model,
    ):
        """
        Re-ranking applied after dedup in pipeline.

        Verifies that re-ranking integrates without errors.
        """
        vector = VectorIndex(embedding_model=shared_embedding_model)
        keyword = KeywordIndex()
        graph = GraphStore()
        manager = IndexManager(config_reranking, vector, keyword, graph)
        orchestrator = SearchOrchestrator(
            vector, keyword, graph, config_reranking, manager
        )

        docs_path = Path(config_reranking.indexing.documents_path)

        # Create documents with varying relevance
        create_test_document(
            docs_path,
            "highly_relevant",
            """# Machine Learning Tutorial

This comprehensive guide covers machine learning fundamentals.
We will explore supervised learning, neural networks, and deep learning.
Machine learning applications include image recognition and NLP.
""",
        )
        create_test_document(
            docs_path,
            "somewhat_relevant",
            """# Data Science Overview

Data science involves statistics and programming.
Machine learning is one component of data science.
""",
        )
        create_test_document(
            docs_path,
            "less_relevant",
            """# Python Programming

Python is a popular programming language.
It is used for web development and scripting.
""",
        )

        for doc_file in docs_path.glob("*.md"):
            manager.index_document(str(doc_file))

        # Query with re-ranking enabled
        results, stats = await orchestrator.query(
            "machine learning tutorial",
            top_k=10,
            top_n=3,
        )

        # Should return results
        assert len(results) > 0
        assert len(results) <= 3

        # Results should be ChunkResult objects
        assert all(isinstance(r, ChunkResult) for r in results)

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_reranking_respects_top_n_config(
        self,
        config_reranking,
        shared_embedding_model,
    ):
        """
        Re-ranking respects rerank_top_n configuration.

        Should return at most rerank_top_n results.
        """
        vector = VectorIndex(embedding_model=shared_embedding_model)
        keyword = KeywordIndex()
        graph = GraphStore()
        manager = IndexManager(config_reranking, vector, keyword, graph)
        orchestrator = SearchOrchestrator(
            vector, keyword, graph, config_reranking, manager
        )

        docs_path = Path(config_reranking.indexing.documents_path)

        # Create multiple documents
        for i in range(10):
            create_test_document(
                docs_path,
                f"doc_{i}",
                f"""# Document {i}

This is document number {i} about testing.
It contains content for search testing purposes.
Document {i} is part of the test corpus.
""",
            )

        for doc_file in docs_path.glob("*.md"):
            manager.index_document(str(doc_file))

        # Query requesting more than rerank_top_n
        results, _ = await orchestrator.query(
            "testing documents",
            top_k=20,
            top_n=10,  # More than rerank_top_n (5)
        )

        # Should be limited by rerank_top_n (5) from config
        assert len(results) <= config_reranking.search.rerank_top_n


# ============================================================================
# Combined Query Expansion and Re-ranking Tests
# ============================================================================


class TestQueryExpansionAndReranking:
    """Integration tests for combined query expansion and re-ranking."""

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_expansion_and_reranking_together(
        self,
        tmp_path,
        shared_embedding_model,
    ):
        """
        Query expansion and re-ranking work together.

        Full pipeline: expand -> search -> fuse -> rerank.
        """
        docs_path = tmp_path / "docs"
        docs_path.mkdir()

        config = Config(
            server=ServerConfig(),
            indexing=IndexingConfig(
                documents_path=str(docs_path),
                index_path=str(tmp_path / "indices"),
            ),
            parsers={"**/*.md": "MarkdownParser"},
            search=SearchConfig(
                semantic_weight=1.0,
                keyword_weight=1.0,
                recency_bias=0.0,
                rrf_k_constant=60,
                rerank_enabled=True,
                rerank_model="cross-encoder/ms-marco-MiniLM-L-6-v2",
                rerank_top_n=3,
                dedup_enabled=False,
            ),
            llm=LLMConfig(embedding_model="BAAI/bge-small-en-v1.5"),
        )

        vector = VectorIndex(embedding_model=shared_embedding_model)
        keyword = KeywordIndex()
        graph = GraphStore()
        manager = IndexManager(config, vector, keyword, graph)
        orchestrator = SearchOrchestrator(vector, keyword, graph, config, manager)

        create_test_document(
            docs_path,
            "auth_guide",
            """# Authentication Guide

Complete guide to authentication and identity verification.
Learn about OAuth, SAML, and other authentication protocols.
Secure your applications with proper authentication.
""",
        )
        create_test_document(
            docs_path,
            "api_docs",
            """# API Reference

REST API endpoints and usage examples.
Authentication required for protected endpoints.
""",
        )

        for doc_file in docs_path.glob("*.md"):
            manager.index_document(str(doc_file))

        # Build vocabulary
        vector.build_concept_vocabulary()

        # Query with short form (expansion) + reranking
        results, stats = await orchestrator.query("auth", top_k=10, top_n=3)

        # Should find results through expansion
        assert len(results) > 0
        # Should be limited by rerank_top_n
        assert len(results) <= 3
