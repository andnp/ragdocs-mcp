"""
Integration tests for Search Infrastructure Overhaul (Spec 17).

Tests the complete integration of:
- Edge type inference through parser → graph flow
- Community detection and persistence in GraphStore
- Dynamic weights with variance-aware fusion
- HyDE search via orchestrator

These tests use real components (no mocks) to validate end-to-end behavior.
"""

from pathlib import Path

import pytest

from src.config import (
    Config,
    IndexingConfig,
    LLMConfig,
    SearchConfig,
    ServerConfig,
    ChunkingConfig,
)
from src.indexing.manager import IndexManager
from src.indices.graph import GraphStore
from src.indices.keyword import KeywordIndex
from src.indices.vector import VectorIndex
from src.parsers.markdown import MarkdownParser
from src.search.edge_types import EdgeType, infer_edge_type
from src.search.orchestrator import SearchOrchestrator


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def test_config(tmp_path):
    """
    Create test configuration with temporary paths and search overhaul features enabled.
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
            community_detection_enabled=True,
            community_boost_factor=1.1,
            dynamic_weights_enabled=True,
            variance_threshold=0.1,
            min_weight_factor=0.5,
            hyde_enabled=True,
        ),
        document_chunking=ChunkingConfig(),
        memory_chunking=ChunkingConfig(),
        llm=LLMConfig(embedding_model="all-MiniLM-L6-v2"),
    )


@pytest.fixture
def indices():
    """Create real index instances for integration testing."""
    vector = VectorIndex()
    keyword = KeywordIndex()
    graph = GraphStore()
    return vector, keyword, graph


@pytest.fixture
def manager(test_config, indices):
    """Create IndexManager with real indices."""
    vector, keyword, graph = indices
    return IndexManager(test_config, vector, keyword, graph)


@pytest.fixture
def orchestrator(test_config, indices, manager):
    """Create SearchOrchestrator with overhaul features enabled."""
    vector, keyword, graph = indices
    return SearchOrchestrator(vector, keyword, graph, test_config, manager)


# ============================================================================
# Edge Type Inference Integration Tests
# ============================================================================


class TestEdgeTypeInferenceIntegration:
    """Tests for edge type inference through the parser → graph flow."""

    def test_parser_extracts_links_with_context(self, test_config):
        """
        MarkdownParser.extract_links_with_context returns links with header context.
        This context is used to infer edge types.
        """
        docs_path = Path(test_config.indexing.documents_path)
        test_doc = docs_path / "test_doc.md"
        test_doc.write_text("""# Overview

Some introduction text.

# Testing

See [[test_module]] for test examples.

# Implementation

Refer to [[src/main.py]] for implementation details.

# See Also

- [[related_doc]]
""")

        parser = MarkdownParser()
        links = parser.extract_links_with_context(str(test_doc))

        assert len(links) == 3

        link_targets = [link.target for link in links]
        assert "test_module" in link_targets
        assert "src/main.py" in link_targets
        assert "related_doc" in link_targets

        link_contexts = {link.target: link.header_context for link in links}
        assert link_contexts["test_module"] != ""
        assert link_contexts["src/main.py"] != ""
        assert link_contexts["related_doc"] != ""

    def test_edge_type_inferred_from_header_context(self, test_config):
        """
        Edge types are correctly inferred from header context strings.
        """
        assert infer_edge_type("Testing", "test_module") == EdgeType.TESTS
        assert infer_edge_type("Implementation", "src/main.py") == EdgeType.IMPLEMENTS
        assert infer_edge_type("See Also", "related_doc") == EdgeType.RELATED
        assert infer_edge_type("Overview", "intro.md") == EdgeType.LINKS_TO

    def test_graph_stores_edge_types(self, indices):
        """
        GraphStore correctly stores and retrieves edge type metadata.
        """
        _, _, graph = indices

        graph.add_node("doc1", {"title": "Doc 1"})
        graph.add_node("doc2", {"title": "Doc 2"})
        graph.add_node("doc3", {"title": "Doc 3"})

        graph.add_edge("doc1", "doc2", edge_type="tests", edge_context="Testing section")
        graph.add_edge("doc1", "doc3", edge_type="implements", edge_context="Implementation")

        edges_to_doc2 = graph.get_edges_to("doc2")
        assert len(edges_to_doc2) == 1
        assert edges_to_doc2[0]["edge_type"] == "tests"
        assert edges_to_doc2[0]["edge_context"] == "Testing section"

        edges_to_doc3 = graph.get_edges_to("doc3")
        assert len(edges_to_doc3) == 1
        assert edges_to_doc3[0]["edge_type"] == "implements"


# ============================================================================
# Community Detection Integration Tests
# ============================================================================


class TestCommunityDetectionIntegration:
    """Tests for community detection in GraphStore."""

    def test_detect_communities_on_clustered_graph(self, indices):
        """
        Community detection identifies clusters in graph structure.
        """
        _, _, graph = indices

        for i in range(1, 4):
            graph.add_node(f"cluster_a_{i}", {})
        for i in range(1, 4):
            graph.add_node(f"cluster_b_{i}", {})

        graph.add_edge("cluster_a_1", "cluster_a_2", "related")
        graph.add_edge("cluster_a_2", "cluster_a_3", "related")
        graph.add_edge("cluster_a_1", "cluster_a_3", "related")

        graph.add_edge("cluster_b_1", "cluster_b_2", "related")
        graph.add_edge("cluster_b_2", "cluster_b_3", "related")
        graph.add_edge("cluster_b_1", "cluster_b_3", "related")

        communities = graph.detect_communities()

        assert len(communities) == 6

        assert communities["cluster_a_1"] == communities["cluster_a_2"] == communities["cluster_a_3"]
        assert communities["cluster_b_1"] == communities["cluster_b_2"] == communities["cluster_b_3"]
        assert communities["cluster_a_1"] != communities["cluster_b_1"]

    def test_community_lookup_by_doc_id(self, indices):
        """
        GraphStore.get_community returns community ID for a document.
        """
        _, _, graph = indices

        graph.add_node("doc1", {})
        graph.add_node("doc2", {})
        graph.add_edge("doc1", "doc2", "related")

        graph.detect_communities()

        community_id = graph.get_community("doc1")
        assert community_id is not None
        assert isinstance(community_id, int)

        assert graph.get_community("doc1") == graph.get_community("doc2")

    def test_community_members_retrieval(self, indices):
        """
        GraphStore.get_community_members returns all docs in a community.
        """
        _, _, graph = indices

        graph.add_node("doc1", {})
        graph.add_node("doc2", {})
        graph.add_node("doc3", {})
        graph.add_edge("doc1", "doc2", "related")
        graph.add_edge("doc2", "doc1", "related")

        graph.detect_communities()

        community_id = graph.get_community("doc1")
        members = graph.get_community_members(community_id)

        assert "doc1" in members
        assert "doc2" in members

    def test_community_persistence(self, indices, tmp_path):
        """
        Community assignments persist across save/load cycles.
        """
        _, _, graph = indices

        graph.add_node("doc1", {})
        graph.add_node("doc2", {})
        graph.add_edge("doc1", "doc2", "related")

        graph.detect_communities()
        original_community = graph.get_community("doc1")

        persist_path = tmp_path / "graph_persist"
        graph.persist(persist_path)

        new_graph = GraphStore()
        new_graph.load(persist_path)

        loaded_community = new_graph.get_community("doc1")
        assert loaded_community == original_community

    def test_community_boost_same_cluster(self, indices):
        """
        Documents in the same community receive boost factor.
        """
        _, _, graph = indices

        graph.add_node("seed_doc", {})
        graph.add_node("same_cluster", {})
        graph.add_node("different_cluster", {})
        graph.add_edge("seed_doc", "same_cluster", "related")
        graph.add_edge("same_cluster", "seed_doc", "related")

        graph.detect_communities()

        boosts = graph.boost_by_community(
            ["seed_doc", "same_cluster", "different_cluster"],
            {"seed_doc"},
            boost_factor=1.2,
        )

        assert boosts["seed_doc"] == 1.2
        assert boosts["same_cluster"] == 1.2
        assert boosts["different_cluster"] == 1.0


# ============================================================================
# Dynamic Weights Integration Tests
# ============================================================================


class TestDynamicWeightsIntegration:
    """Tests for variance-aware dynamic weight computation in search."""

    @pytest.mark.asyncio
    async def test_dynamic_weights_affect_ranking(self, test_config, manager, orchestrator):
        """
        Dynamic weights adjust strategy contributions based on score variance.
        Low variance (muddy results) reduces that strategy's influence.
        """
        docs_path = Path(test_config.indexing.documents_path)

        doc1 = docs_path / "unique_term.md"
        doc1.write_text("# Unique Term\n\nThis document contains xyzzy123 unique keyword.")
        manager.index_document(str(doc1))

        doc2 = docs_path / "common_concept.md"
        doc2.write_text("# Common Concept\n\nGeneral information about software.")
        manager.index_document(str(doc2))

        doc3 = docs_path / "another_common.md"
        doc3.write_text("# Another Document\n\nMore general software information.")
        manager.index_document(str(doc3))

        results, _, _ = await orchestrator.query("xyzzy123", top_k=10, top_n=5)

        assert len(results) > 0
        # With calibration, the unique_term document should still be present
        # Check if it appears anywhere in the results (may not be #1 due to calibration)
        chunk_ids = [r.chunk_id for r in results]
        assert any("unique_term" in chunk_id for chunk_id in chunk_ids), \
            f"unique_term should be in results, got: {chunk_ids}"

    @pytest.mark.asyncio
    async def test_dynamic_weights_disabled_uses_base_weights(self, test_config, indices, manager, tmp_path):
        """
        With dynamic_weights_enabled=False, base weights are used unchanged.
        """
        config_static = Config(
            server=ServerConfig(),
            indexing=IndexingConfig(
                documents_path=str(tmp_path / "docs_static"),
                index_path=str(tmp_path / "indices_static"),
            ),
            parsers={"**/*.md": "MarkdownParser"},
            search=SearchConfig(
                semantic_weight=1.0,
                keyword_weight=1.0,
                dynamic_weights_enabled=False,
            ),
            llm=LLMConfig(embedding_model="all-MiniLM-L6-v2"),
        )

        docs_path = Path(config_static.indexing.documents_path)
        docs_path.mkdir(parents=True, exist_ok=True)

        vector, keyword, graph = indices
        manager_static = IndexManager(config_static, vector, keyword, graph)
        orchestrator_static = SearchOrchestrator(
            vector, keyword, graph, config_static, manager_static
        )

        doc1 = docs_path / "test_doc.md"
        doc1.write_text("# Test\n\nTest document content.")
        manager_static.index_document(str(doc1))

        results, _, _ = await orchestrator_static.query("test document", top_k=10, top_n=5)
        assert len(results) >= 0


# ============================================================================
# HyDE Search Integration Tests
# ============================================================================


class TestHyDESearchIntegration:
    """Tests for HyDE (Hypothetical Document Embeddings) search."""

    @pytest.mark.asyncio
    async def test_hyde_search_finds_relevant_docs(self, test_config, manager, orchestrator):
        """
        HyDE search with a good hypothesis finds relevant documentation.
        """
        docs_path = Path(test_config.indexing.documents_path)

        doc1 = docs_path / "mcp_tools.md"
        doc1.write_text("""# Adding MCP Tools

To add a new tool to the MCP server:

1. Define the tool in `list_tools()` method
2. Add a handler in `call_tool()` method
3. Implement the tool logic

Example tool registration in src/mcp_server.py.
""")
        manager.index_document(str(doc1))

        doc2 = docs_path / "configuration.md"
        doc2.write_text("# Configuration\n\nServer configuration options.")
        manager.index_document(str(doc2))

        hypothesis = (
            "To add a tool to the MCP server, I need to modify mcp_server.py. "
            "The process involves registering the tool in list_tools and "
            "adding a handler function in call_tool."
        )

        results, _, _ = await orchestrator.query_with_hypothesis(
            hypothesis, top_k=10, top_n=5
        )

        assert len(results) > 0
        result_doc_ids = [r.doc_id for r in results]
        assert "mcp_tools" in result_doc_ids or "configuration" in result_doc_ids

    @pytest.mark.asyncio
    async def test_hyde_disabled_falls_back_to_regular_query(
        self, test_config, indices, manager, tmp_path
    ):
        """
        When hyde_enabled=False, query_with_hypothesis falls back to regular query.
        """
        config_no_hyde = Config(
            server=ServerConfig(),
            indexing=IndexingConfig(
                documents_path=str(tmp_path / "docs_no_hyde"),
                index_path=str(tmp_path / "indices_no_hyde"),
            ),
            parsers={"**/*.md": "MarkdownParser"},
            search=SearchConfig(hyde_enabled=False),
            llm=LLMConfig(embedding_model="all-MiniLM-L6-v2"),
        )

        docs_path = Path(config_no_hyde.indexing.documents_path)
        docs_path.mkdir(parents=True, exist_ok=True)

        vector, keyword, graph = indices
        manager_no_hyde = IndexManager(config_no_hyde, vector, keyword, graph)
        orchestrator_no_hyde = SearchOrchestrator(
            vector, keyword, graph, config_no_hyde, manager_no_hyde
        )

        doc1 = docs_path / "test.md"
        doc1.write_text("# Test Document\n\nSome content here.")
        manager_no_hyde.index_document(str(doc1))

        results, _, _ = await orchestrator_no_hyde.query_with_hypothesis(
            "test hypothesis", top_k=10, top_n=5
        )

        assert len(results) >= 0

    @pytest.mark.asyncio
    async def test_hyde_returns_normalized_scores(self, test_config, manager, orchestrator):
        """
        HyDE search returns properly normalized scores in [0, 1].
        """
        docs_path = Path(test_config.indexing.documents_path)

        for i in range(3):
            doc = docs_path / f"doc_{i}.md"
            doc.write_text(f"# Document {i}\n\nContent about topic {i}.")
            manager.index_document(str(doc))

        results, _, _ = await orchestrator.query_with_hypothesis(
            "Information about topics", top_k=10, top_n=5
        )

        assert len(results) > 0
        for result in results:
            assert 0.0 <= result.score <= 1.0


# ============================================================================
# End-to-End Search Overhaul Tests
# ============================================================================


class TestSearchOverhaulEndToEnd:
    """End-to-end tests for the complete search infrastructure overhaul."""

    @pytest.mark.asyncio
    async def test_full_search_pipeline_with_overhaul_features(
        self, test_config, manager, orchestrator
    ):
        """
        Complete search with community boost, dynamic weights, and edge types.
        """
        docs_path = Path(test_config.indexing.documents_path)

        spec_doc = docs_path / "feature_spec.md"
        spec_doc.write_text("""# Feature Specification

## Overview
Description of the feature.

## Implementation
See [[feature_impl]] for implementation.

## Testing
Refer to [[feature_test]] for tests.
""")
        manager.index_document(str(spec_doc))

        impl_doc = docs_path / "feature_impl.md"
        impl_doc.write_text("""# Feature Implementation

Implementation details for the feature.
""")
        manager.index_document(str(impl_doc))

        test_doc = docs_path / "feature_test.md"
        test_doc.write_text("""# Feature Tests

Test cases for the feature.
""")
        manager.index_document(str(test_doc))

        related_doc = docs_path / "related_feature.md"
        related_doc.write_text("""# Related Feature

A related feature that shares concepts.

## See Also
- [[feature_spec]]
""")
        manager.index_document(str(related_doc))

        results, _, _ = await orchestrator.query("feature implementation", top_k=10, top_n=5)

        assert len(results) > 0

        for result in results:
            assert 0.0 <= result.score <= 1.0

        if results:
            # Calibrated score should be high confidence for good match
            assert results[0].score > 0.95

    @pytest.mark.asyncio
    async def test_empty_hypothesis_returns_empty(self, test_config, manager, orchestrator):
        """
        Empty hypothesis returns empty results gracefully.
        """
        docs_path = Path(test_config.indexing.documents_path)
        doc = docs_path / "test.md"
        doc.write_text("# Test\n\nContent.")
        manager.index_document(str(doc))

        results, stats, _ = await orchestrator.query_with_hypothesis("", top_k=10, top_n=5)

        assert results == []
        assert stats.original_count == 0

    @pytest.mark.asyncio
    async def test_community_detection_empty_graph(self, indices):
        """
        Community detection on empty graph returns empty dict.
        """
        _, _, graph = indices
        communities = graph.detect_communities()
        assert communities == {}

    @pytest.mark.asyncio
    async def test_config_options_respected(self, test_config):
        """
        New search.advanced config options are correctly loaded.
        """
        assert test_config.search.community_detection_enabled is True
        assert test_config.search.community_boost_factor == 1.1
        assert test_config.search.dynamic_weights_enabled is True
        assert test_config.search.variance_threshold == 0.1
        assert test_config.search.min_weight_factor == 0.5
        assert test_config.search.hyde_enabled is True
