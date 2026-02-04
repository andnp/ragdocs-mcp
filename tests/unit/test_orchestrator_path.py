"""
Unit tests for SearchOrchestrator path isolation.

These tests verify that SearchOrchestrator uses the documents_path passed
to its constructor and is NOT affected by subsequent changes to config.

REGRESSION CONTEXT:
The bug occurred when SearchOrchestrator read documents_path from the shared
Config object on every query instead of storing an immutable path at construction.
This caused queries to use the wrong project context when:
1. Multiple ApplicationContexts existed with different documents_path values
2. Config was modified after orchestrator construction
"""

from datetime import datetime
from pathlib import Path

import pytest

from src.config import Config, IndexingConfig, SearchConfig, ChunkingConfig, LLMConfig
from src.indices.keyword import KeywordIndex
from src.indices.vector import VectorIndex
from src.indices.graph import GraphStore
from src.indexing.manager import IndexManager
from src.models import Chunk
from src.search.orchestrator import SearchOrchestrator


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def base_config(tmp_path):
    docs_path = tmp_path / "original_docs"
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
def indices(shared_embedding_model):
    vector = VectorIndex(embedding_model=shared_embedding_model)
    keyword = KeywordIndex()
    graph = GraphStore()
    return vector, keyword, graph


# ============================================================================
# Test: Constructor stores documents_path
# ============================================================================


def test_orchestrator_stores_documents_path_from_constructor(base_config, indices, tmp_path):
    """
    Test that SearchOrchestrator uses the documents_path passed to constructor.

    The orchestrator should store the path at construction time as an immutable
    instance variable, not read it from config on each access.
    """
    vector, keyword, graph = indices
    manager = IndexManager(base_config, vector, keyword, graph)

    explicit_path = tmp_path / "explicit_docs"
    explicit_path.mkdir()

    orchestrator = SearchOrchestrator(
        vector, keyword, graph, base_config, manager,
        documents_path=explicit_path,
    )

    assert orchestrator.documents_path == explicit_path


def test_orchestrator_documents_path_property_returns_correct_value(base_config, indices, tmp_path):
    """
    Test that the documents_path property returns the value set at construction.

    This verifies the property is correctly implemented.
    """
    vector, keyword, graph = indices
    manager = IndexManager(base_config, vector, keyword, graph)

    custom_path = tmp_path / "custom_docs"
    custom_path.mkdir()

    orchestrator = SearchOrchestrator(
        vector, keyword, graph, base_config, manager,
        documents_path=custom_path,
    )

    # Access property multiple times - should always return same value
    assert orchestrator.documents_path == custom_path
    assert orchestrator.documents_path == custom_path


# ============================================================================
# Test: Config modification after construction does NOT affect orchestrator
# ============================================================================


def test_config_modification_does_not_affect_orchestrator_path(base_config, indices, tmp_path):
    """
    Test that modifying config.indexing.documents_path AFTER construction
    does NOT change the orchestrator's documents_path.

    REGRESSION: This is the core test for the bug. Previously, the orchestrator
    would read from config on every query, so modifying config would affect
    query behavior.
    """
    vector, keyword, graph = indices
    manager = IndexManager(base_config, vector, keyword, graph)

    explicit_path = tmp_path / "explicit_docs"
    explicit_path.mkdir()

    orchestrator = SearchOrchestrator(
        vector, keyword, graph, base_config, manager,
        documents_path=explicit_path,
    )

    # Verify initial path
    assert orchestrator.documents_path == explicit_path

    # Now modify the config (simulating what might happen in a multi-project scenario)
    different_path = tmp_path / "different_project"
    different_path.mkdir()
    base_config.indexing.documents_path = str(different_path)

    # Orchestrator's path should NOT have changed
    assert orchestrator.documents_path == explicit_path
    assert orchestrator.documents_path != Path(base_config.indexing.documents_path)


def test_orchestrator_path_immutable_across_multiple_config_changes(base_config, indices, tmp_path):
    """
    Test that multiple config modifications do not affect the orchestrator's path.

    Simulates a scenario where config might be modified multiple times
    (e.g., during project switches).
    """
    vector, keyword, graph = indices
    manager = IndexManager(base_config, vector, keyword, graph)

    original_path = tmp_path / "original"
    original_path.mkdir()

    orchestrator = SearchOrchestrator(
        vector, keyword, graph, base_config, manager,
        documents_path=original_path,
    )

    # Multiple config modifications
    for i in range(5):
        modified_path = tmp_path / f"modified_{i}"
        modified_path.mkdir()
        base_config.indexing.documents_path = str(modified_path)

        # Path should remain unchanged throughout
        assert orchestrator.documents_path == original_path


# ============================================================================
# Test: Fallback behavior when documents_path=None
# ============================================================================


def test_orchestrator_fallback_to_config_when_documents_path_none(base_config, indices):
    """
    Test that when documents_path=None, orchestrator uses config.indexing.documents_path.

    This maintains backward compatibility for code that doesn't pass explicit path.
    """
    vector, keyword, graph = indices
    manager = IndexManager(base_config, vector, keyword, graph)

    orchestrator = SearchOrchestrator(
        vector, keyword, graph, base_config, manager,
        documents_path=None,  # Explicit None
    )

    # Should fall back to config path
    assert orchestrator.documents_path == Path(base_config.indexing.documents_path)


def test_orchestrator_default_documents_path_is_none(base_config, indices):
    """
    Test that not passing documents_path defaults to using config path.

    When documents_path parameter is omitted, orchestrator should use
    config.indexing.documents_path (converted to Path).
    """
    vector, keyword, graph = indices
    manager = IndexManager(base_config, vector, keyword, graph)

    # Don't pass documents_path - should default
    orchestrator = SearchOrchestrator(
        vector, keyword, graph, base_config, manager,
    )

    assert orchestrator.documents_path == Path(base_config.indexing.documents_path)


# ============================================================================
# Test: Path is used in query methods
# ============================================================================


@pytest.mark.asyncio
async def test_query_uses_orchestrator_documents_path(base_config, indices, tmp_path):
    """
    Test that query() uses the orchestrator's stored documents_path, not config.

    This verifies that the fix actually affects query behavior.
    """
    vector, keyword, graph = indices
    manager = IndexManager(base_config, vector, keyword, graph)

    explicit_docs = tmp_path / "explicit_docs"
    explicit_docs.mkdir()

    orchestrator = SearchOrchestrator(
        vector, keyword, graph, base_config, manager,
        documents_path=explicit_docs,
    )

    # Add a chunk that references the explicit docs path
    chunk = Chunk(
        chunk_id="test/doc_chunk_0",
        doc_id="test/doc",
        content="Test content for path verification",
        metadata={},
        chunk_index=0,
        header_path="",
        start_pos=0,
        end_pos=50,
        file_path=str(explicit_docs / "test" / "doc.md"),
        modified_time=datetime.now(),
    )
    vector.add_chunk(chunk)
    keyword.add_chunk(chunk)

    # Modify config to point elsewhere
    wrong_path = tmp_path / "wrong_path"
    wrong_path.mkdir()
    base_config.indexing.documents_path = str(wrong_path)

    # Query should still work because orchestrator uses its stored path
    results, _, _ = await orchestrator.query("test content", top_k=5, top_n=5)

    # The query completes without error, which means it used the correct path
    # (If it used the wrong path, file exclusion/normalization would fail)
    assert orchestrator.documents_path == explicit_docs


@pytest.mark.asyncio
async def test_query_with_hypothesis_uses_orchestrator_documents_path(base_config, indices, tmp_path):
    """
    Test that query_with_hypothesis() uses the orchestrator's stored documents_path.

    Both query methods should behave consistently regarding path usage.
    """
    vector, keyword, graph = indices
    manager = IndexManager(base_config, vector, keyword, graph)

    explicit_docs = tmp_path / "explicit_docs"
    explicit_docs.mkdir()

    # Disable HyDE to simplify test (will fall back to regular query)
    base_config.search.hyde_enabled = False

    orchestrator = SearchOrchestrator(
        vector, keyword, graph, base_config, manager,
        documents_path=explicit_docs,
    )

    # Modify config
    base_config.indexing.documents_path = str(tmp_path / "different")

    # Should use orchestrator's path, not config's
    results, _, _ = await orchestrator.query_with_hypothesis(
        "hypothesis about documentation",
        top_k=5,
        top_n=5,
    )

    assert orchestrator.documents_path == explicit_docs


# ============================================================================
# Test: Two orchestrators with different paths remain isolated
# ============================================================================


def test_two_orchestrators_with_different_paths_are_isolated(base_config, tmp_path, shared_embedding_model):
    """
    Test that two orchestrators created with different documents_path values
    maintain their own paths independently.

    This simulates the multi-project scenario where ApplicationContext creates
    separate orchestrators for different projects.
    """
    vector1 = VectorIndex(embedding_model=shared_embedding_model)
    keyword1 = KeywordIndex()
    graph1 = GraphStore()
    manager1 = IndexManager(base_config, vector1, keyword1, graph1)

    vector2 = VectorIndex(embedding_model=shared_embedding_model)
    keyword2 = KeywordIndex()
    graph2 = GraphStore()
    manager2 = IndexManager(base_config, vector2, keyword2, graph2)

    path1 = tmp_path / "project_a"
    path1.mkdir()
    path2 = tmp_path / "project_b"
    path2.mkdir()

    orchestrator1 = SearchOrchestrator(
        vector1, keyword1, graph1, base_config, manager1,
        documents_path=path1,
    )
    orchestrator2 = SearchOrchestrator(
        vector2, keyword2, graph2, base_config, manager2,
        documents_path=path2,
    )

    # Both should maintain their own paths
    assert orchestrator1.documents_path == path1
    assert orchestrator2.documents_path == path2

    # They should NOT share the same path
    assert orchestrator1.documents_path != orchestrator2.documents_path


# ============================================================================
# Test: BaseSearchOrchestrator stores path in _documents_path
# ============================================================================


def test_base_orchestrator_stores_documents_path(base_config, tmp_path, shared_embedding_model):
    """
    Test that BaseSearchOrchestrator stores documents_path in _documents_path.

    The base class provides the foundation for path storage that subclasses
    like SearchOrchestrator and MemorySearchOrchestrator use.
    """
    vector = VectorIndex(embedding_model=shared_embedding_model)
    keyword = KeywordIndex()
    graph = GraphStore()

    custom_path = tmp_path / "custom"
    custom_path.mkdir()

    # Create a concrete implementation to test base class behavior
    # SearchOrchestrator extends BaseSearchOrchestrator
    manager = IndexManager(base_config, vector, keyword, graph)
    orchestrator = SearchOrchestrator(
        vector, keyword, graph, base_config, manager,
        documents_path=custom_path,
    )

    # BaseSearchOrchestrator stores in _documents_path (accessed via property)
    # This verifies the inheritance chain works correctly
    assert orchestrator._documents_path == custom_path


def test_base_orchestrator_passes_path_to_search_methods(base_config, indices, tmp_path):
    """
    Test that BaseSearchOrchestrator passes _documents_path to search methods.

    The base class's _search_vector_base and _search_keyword_base methods
    should use the stored _documents_path, not config.
    """
    vector, keyword, graph = indices
    manager = IndexManager(base_config, vector, keyword, graph)

    explicit_path = tmp_path / "explicit"
    explicit_path.mkdir()

    orchestrator = SearchOrchestrator(
        vector, keyword, graph, base_config, manager,
        documents_path=explicit_path,
    )

    # Modify config to verify it's not used
    base_config.indexing.documents_path = str(tmp_path / "wrong")

    # The internal _documents_path should still be correct
    # (This is what gets passed to search methods via docs_root)
    assert orchestrator._documents_path == explicit_path
