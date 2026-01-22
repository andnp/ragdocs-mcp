"""
Integration tests for multi-project context isolation.

These tests verify that when multiple ApplicationContexts are created for
different projects, each context's orchestrator uses the correct documents_path
and returns results only from its own project.

REGRESSION CONTEXT:
The bug occurred when SearchOrchestrator read documents_path from the shared
Config object on every query. This caused queries on one project's context
to potentially return results from a different project if Config was modified.
"""

from datetime import datetime
from pathlib import Path

import pytest

from src.config import Config, IndexingConfig, SearchConfig, ChunkingConfig, LLMConfig, ServerConfig
from src.context import ApplicationContext
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
def two_projects(tmp_path):
    """
    Create two separate project directories with different documents.
    """
    project_a = tmp_path / "project_a"
    project_a_docs = project_a / "docs"
    project_a_docs.mkdir(parents=True)
    (project_a_docs / "readme.md").write_text("# Project A\n\nThis is Project Alpha documentation.")
    (project_a_docs / "api.md").write_text("# Project A API\n\nProject Alpha API reference.")

    project_b = tmp_path / "project_b"
    project_b_docs = project_b / "docs"
    project_b_docs.mkdir(parents=True)
    (project_b_docs / "readme.md").write_text("# Project B\n\nThis is Project Beta documentation.")
    (project_b_docs / "guide.md").write_text("# Project B Guide\n\nProject Beta user guide.")

    return {
        "project_a": project_a,
        "project_a_docs": project_a_docs,
        "project_b": project_b,
        "project_b_docs": project_b_docs,
        "tmp": tmp_path,
    }


@pytest.fixture
def config_for_project_a(two_projects, tmp_path):
    return Config(
        server=ServerConfig(),
        indexing=IndexingConfig(
            documents_path=str(two_projects["project_a_docs"]),
            index_path=str(tmp_path / "index_a"),
        ),
        search=SearchConfig(),
        document_chunking=ChunkingConfig(),
        memory_chunking=ChunkingConfig(),
        llm=LLMConfig(embedding_model="BAAI/bge-small-en-v1.5"),
    )


@pytest.fixture
def config_for_project_b(two_projects, tmp_path):
    return Config(
        server=ServerConfig(),
        indexing=IndexingConfig(
            documents_path=str(two_projects["project_b_docs"]),
            index_path=str(tmp_path / "index_b"),
        ),
        search=SearchConfig(),
        document_chunking=ChunkingConfig(),
        memory_chunking=ChunkingConfig(),
        llm=LLMConfig(embedding_model="BAAI/bge-small-en-v1.5"),
    )


# ============================================================================
# Test: Two orchestrators maintain separate paths
# ============================================================================


def test_orchestrators_for_different_projects_have_different_paths(
    config_for_project_a,
    config_for_project_b,
    two_projects,
):
    """
    Test that orchestrators created for different projects maintain separate paths.

    Each orchestrator should use the documents_path from its own config,
    not share a global state.
    """
    # Create orchestrator for Project A
    vector_a = VectorIndex(embedding_model_name="BAAI/bge-small-en-v1.5")
    keyword_a = KeywordIndex()
    graph_a = GraphStore()
    manager_a = IndexManager(config_for_project_a, vector_a, keyword_a, graph_a)
    orchestrator_a = SearchOrchestrator(
        vector_a, keyword_a, graph_a, config_for_project_a, manager_a,
        documents_path=Path(config_for_project_a.indexing.documents_path),
    )

    # Create orchestrator for Project B
    vector_b = VectorIndex(embedding_model_name="BAAI/bge-small-en-v1.5")
    keyword_b = KeywordIndex()
    graph_b = GraphStore()
    manager_b = IndexManager(config_for_project_b, vector_b, keyword_b, graph_b)
    orchestrator_b = SearchOrchestrator(
        vector_b, keyword_b, graph_b, config_for_project_b, manager_b,
        documents_path=Path(config_for_project_b.indexing.documents_path),
    )

    # Verify paths are different
    assert orchestrator_a.documents_path != orchestrator_b.documents_path
    assert orchestrator_a.documents_path == two_projects["project_a_docs"]
    assert orchestrator_b.documents_path == two_projects["project_b_docs"]


# ============================================================================
# Test: Queries return results from correct project
# ============================================================================


@pytest.mark.asyncio
async def test_queries_return_results_from_correct_project(
    config_for_project_a,
    config_for_project_b,
    two_projects,
):
    """
    Test that querying each orchestrator returns results only from its project.

    REGRESSION: Previously, if config was shared/modified, queries could return
    results from the wrong project. This test verifies isolation.
    """
    # Create and populate orchestrator for Project A
    vector_a = VectorIndex(embedding_model_name="BAAI/bge-small-en-v1.5")
    keyword_a = KeywordIndex()
    graph_a = GraphStore()
    manager_a = IndexManager(config_for_project_a, vector_a, keyword_a, graph_a)
    orchestrator_a = SearchOrchestrator(
        vector_a, keyword_a, graph_a, config_for_project_a, manager_a,
        documents_path=Path(config_for_project_a.indexing.documents_path),
    )

    # Add Project A specific chunk
    chunk_a = Chunk(
        chunk_id="readme_chunk_0",
        doc_id="readme",
        content="Project Alpha documentation",
        metadata={},
        chunk_index=0,
        header_path="Project A",
        start_pos=0,
        end_pos=30,
        file_path=str(two_projects["project_a_docs"] / "readme.md"),
        modified_time=datetime.now(),
    )
    vector_a.add_chunk(chunk_a)
    keyword_a.add_chunk(chunk_a)

    # Create and populate orchestrator for Project B
    vector_b = VectorIndex(embedding_model_name="BAAI/bge-small-en-v1.5")
    keyword_b = KeywordIndex()
    graph_b = GraphStore()
    manager_b = IndexManager(config_for_project_b, vector_b, keyword_b, graph_b)
    orchestrator_b = SearchOrchestrator(
        vector_b, keyword_b, graph_b, config_for_project_b, manager_b,
        documents_path=Path(config_for_project_b.indexing.documents_path),
    )

    # Add Project B specific chunk
    chunk_b = Chunk(
        chunk_id="readme_chunk_0",
        doc_id="readme",
        content="Project Beta documentation",
        metadata={},
        chunk_index=0,
        header_path="Project B",
        start_pos=0,
        end_pos=30,
        file_path=str(two_projects["project_b_docs"] / "readme.md"),
        modified_time=datetime.now(),
    )
    vector_b.add_chunk(chunk_b)
    keyword_b.add_chunk(chunk_b)

    # Query Project A for "Alpha" - should find results
    results_a, _, _ = await orchestrator_a.query("Alpha", top_k=5, top_n=5)
    assert len(results_a) > 0
    assert any("Alpha" in r.content for r in results_a)

    # Query Project B for "Beta" - should find results
    results_b, _, _ = await orchestrator_b.query("Beta", top_k=5, top_n=5)
    assert len(results_b) > 0
    assert any("Beta" in r.content for r in results_b)

    # Cross-check: Query Project A for "Beta" - should NOT find results
    results_a_beta, _, _ = await orchestrator_a.query("Beta", top_k=5, top_n=5)
    assert all("Beta" not in r.content for r in results_a_beta)

    # Cross-check: Query Project B for "Alpha" - should NOT find results
    results_b_alpha, _, _ = await orchestrator_b.query("Alpha", top_k=5, top_n=5)
    assert all("Alpha" not in r.content for r in results_b_alpha)


# ============================================================================
# Test: ApplicationContext creates orchestrator with correct path
# ============================================================================


def test_application_context_creates_orchestrator_with_correct_path(
    config_for_project_a,
    two_projects,
    monkeypatch,
):
    """
    Test that ApplicationContext.create() passes the correct documents_path
    to the SearchOrchestrator.

    This verifies the integration between context creation and orchestrator setup.
    """
    monkeypatch.setattr("src.context.load_config", lambda: config_for_project_a)

    ctx = ApplicationContext.create(
        project_override=None,
        enable_watcher=False,
        lazy_embeddings=True,
    )

    # Orchestrator should have the correct documents_path
    assert ctx.orchestrator.documents_path == Path(config_for_project_a.indexing.documents_path)


def test_multiple_contexts_have_isolated_orchestrator_paths(
    config_for_project_a,
    config_for_project_b,
    two_projects,
    monkeypatch,
):
    """
    Test that creating multiple ApplicationContexts for different projects
    results in each having an isolated orchestrator with the correct path.

    REGRESSION: This is the key multi-project isolation test. Previously,
    orchestrators could share the same path if they read from a shared config.
    """
    # Create context for Project A
    monkeypatch.setattr("src.context.load_config", lambda: config_for_project_a)
    ctx_a = ApplicationContext.create(
        project_override=None,
        enable_watcher=False,
        lazy_embeddings=True,
    )

    # Create context for Project B (need to update monkeypatch)
    monkeypatch.setattr("src.context.load_config", lambda: config_for_project_b)
    ctx_b = ApplicationContext.create(
        project_override=None,
        enable_watcher=False,
        lazy_embeddings=True,
    )

    # Verify paths are different and correct
    assert ctx_a.orchestrator.documents_path != ctx_b.orchestrator.documents_path
    assert ctx_a.orchestrator.documents_path == Path(config_for_project_a.indexing.documents_path)
    assert ctx_b.orchestrator.documents_path == Path(config_for_project_b.indexing.documents_path)


# ============================================================================
# Test: Config modification doesn't affect existing contexts
# ============================================================================


def test_config_modification_does_not_affect_existing_context(
    config_for_project_a,
    two_projects,
    monkeypatch,
):
    """
    Test that modifying config after context creation does not affect
    the existing orchestrator's documents_path.

    This simulates scenarios where config might be modified after initialization.
    """
    monkeypatch.setattr("src.context.load_config", lambda: config_for_project_a)

    ctx = ApplicationContext.create(
        project_override=None,
        enable_watcher=False,
        lazy_embeddings=True,
    )

    original_path = ctx.orchestrator.documents_path

    # Modify the config
    config_for_project_a.indexing.documents_path = str(two_projects["project_b_docs"])

    # Orchestrator's path should NOT have changed
    assert ctx.orchestrator.documents_path == original_path
    assert ctx.orchestrator.documents_path == Path(two_projects["project_a_docs"])


# ============================================================================
# Test: File exclusions resolve against orchestrator path
# ============================================================================


@pytest.mark.asyncio
async def test_file_exclusions_resolve_against_orchestrator_path(
    config_for_project_a,
    two_projects,
):
    """
    Test that excluded_files in queries resolve against the orchestrator's
    documents_path, not the config's current value.

    This ensures file exclusion works correctly in multi-project scenarios.
    """
    vector = VectorIndex(embedding_model_name="BAAI/bge-small-en-v1.5")
    keyword = KeywordIndex()
    graph = GraphStore()
    manager = IndexManager(config_for_project_a, vector, keyword, graph)

    orchestrator = SearchOrchestrator(
        vector, keyword, graph, config_for_project_a, manager,
        documents_path=Path(config_for_project_a.indexing.documents_path),
    )

    # Add multiple chunks
    for name in ["readme", "api", "guide"]:
        chunk = Chunk(
            chunk_id=f"{name}_chunk_0",
            doc_id=name,
            content=f"Content from {name} file in Project A",
            metadata={},
            chunk_index=0,
            header_path=name,
            start_pos=0,
            end_pos=50,
            file_path=str(two_projects["project_a_docs"] / f"{name}.md"),
            modified_time=datetime.now(),
        )
        vector.add_chunk(chunk)
        keyword.add_chunk(chunk)

    # Change config to point elsewhere (simulating project switch)
    config_for_project_a.indexing.documents_path = str(two_projects["project_b_docs"])

    # Exclusion should still work because it uses orchestrator.documents_path
    # The file path "readme" should resolve against Project A's docs path
    results, _, _ = await orchestrator.query(
        "Content Project",
        top_k=10,
        top_n=10,
        excluded_files={"readme"},
    )

    # Should have results but NOT from readme
    chunk_ids = [r.chunk_id for r in results]
    assert "readme_chunk_0" not in chunk_ids


# ============================================================================
# Test: Persistence and reload maintain correct paths
# ============================================================================


@pytest.mark.asyncio
async def test_context_persistence_maintains_path_isolation(
    config_for_project_a,
    two_projects,
    tmp_path,
    monkeypatch,
):
    """
    Test that after persisting and reloading indices, the orchestrator
    still uses the correct documents_path.
    """
    monkeypatch.setattr("src.context.load_config", lambda: config_for_project_a)

    # Create and start context
    ctx = ApplicationContext.create(
        project_override=None,
        enable_watcher=False,
        lazy_embeddings=True,
    )

    original_docs_path = ctx.orchestrator.documents_path

    await ctx.start(background_index=False)

    # After start, path should still be correct
    assert ctx.orchestrator.documents_path == original_docs_path

    await ctx.stop()

    # Path should remain correct even after stop
    assert ctx.orchestrator.documents_path == original_docs_path
