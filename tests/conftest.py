"""
Shared pytest fixtures for integration and e2e tests.

Provides both ephemeral (tmp_path) and persistent fixtures for different
testing scenarios:

- Ephemeral fixtures (tmp_path): Fast, isolated, used by default in unit tests
- Persistent fixtures: Realistic storage, shared across tests in a session/module

Use persistent fixtures when:
- Testing index persistence/loading behavior
- Testing manifest checking across test runs
- Simulating realistic production scenarios
- Testing index size/performance with larger datasets

Use ephemeral fixtures (tmp_path) when:
- Testing core logic in isolation
- Fast test iteration is priority
- Each test needs complete isolation
"""

from pathlib import Path
from typing import Generator

import pytest

from src.config import Config, IndexingConfig, LLMConfig, SearchConfig, ServerConfig
from src.indexing.manager import IndexManager
from src.indices.graph import GraphStore
from src.indices.keyword import KeywordIndex
from src.indices.vector import VectorIndex


# ============================================================================
# Persistent Storage Fixtures
# ============================================================================


@pytest.fixture(scope="session")
def persistent_storage_root(tmp_path_factory) -> Path:
    """
    Create session-scoped persistent storage directory.

    This directory persists for the entire test session, allowing
    tests to share data and verify persistence behavior.

    Returns path to persistent storage root directory.
    """
    return tmp_path_factory.mktemp("persistent_test_storage")


@pytest.fixture(scope="session")
def persistent_docs_path(persistent_storage_root: Path) -> Path:
    """
    Create session-scoped documents directory.

    Documents stored here persist across tests in the session.

    Returns path to persistent documents directory.
    """
    docs_path = persistent_storage_root / "documents"
    docs_path.mkdir(parents=True, exist_ok=True)
    return docs_path


@pytest.fixture(scope="session")
def persistent_index_path(persistent_storage_root: Path) -> Path:
    """
    Create session-scoped index directory.

    Indices stored here persist across tests in the session.

    Returns path to persistent index directory.
    """
    index_path = persistent_storage_root / "indices"
    index_path.mkdir(parents=True, exist_ok=True)
    return index_path


# ============================================================================
# Persistent Configuration Fixtures
# ============================================================================


@pytest.fixture(scope="session")
def persistent_config(
    persistent_docs_path: Path,
    persistent_index_path: Path,
) -> Config:
    """
    Create session-scoped configuration with persistent paths.

    Uses real persistent storage locations that survive across
    tests in the session.

    Returns Config object configured for persistent storage.
    """
    return Config(
        server=ServerConfig(),
        indexing=IndexingConfig(
            documents_path=str(persistent_docs_path),
            index_path=str(persistent_index_path),
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
    )


# ============================================================================
# Module-Scoped Persistent Fixtures
# ============================================================================


@pytest.fixture(scope="module")
def persistent_indices_module() -> Generator[tuple[VectorIndex, KeywordIndex, GraphStore], None, None]:
    """
    Create module-scoped indices that persist across tests in a module.

    These indices are shared across all tests in a module for performance.
    They start fresh but can accumulate data within a module's test suite.

    Yields tuple of (vector, keyword, graph) indices.
    """
    vector = VectorIndex()
    keyword = KeywordIndex()
    graph = GraphStore()
    yield vector, keyword, graph


@pytest.fixture(scope="module")
def persistent_manager_module(
    persistent_config: Config,
    persistent_indices_module: tuple[VectorIndex, KeywordIndex, GraphStore],
) -> IndexManager:
    """
    Create module-scoped IndexManager with persistent storage.

    This manager uses persistent paths and shared indices within a module.
    Useful for testing persistence behavior and manifest checking.

    Returns IndexManager configured with persistent storage.
    """
    vector, keyword, graph = persistent_indices_module
    return IndexManager(persistent_config, vector, keyword, graph)


# ============================================================================
# Function-Scoped Persistent Fixtures with Cleanup
# ============================================================================


@pytest.fixture
def persistent_indices_isolated() -> Generator[tuple[VectorIndex, KeywordIndex, GraphStore], None, None]:
    """
    Create function-scoped indices that can use persistent storage.

    Fresh indices for each test but can persist to/load from disk.
    Provides isolation while allowing persistence testing.

    Yields tuple of (vector, keyword, graph) indices.
    """
    vector = VectorIndex()
    keyword = KeywordIndex()
    graph = GraphStore()
    yield vector, keyword, graph


@pytest.fixture
def persistent_manager_isolated(
    persistent_config: Config,
    persistent_indices_isolated: tuple[VectorIndex, KeywordIndex, GraphStore],
) -> IndexManager:
    """
    Create function-scoped IndexManager with persistent storage.

    Fresh manager for each test that uses persistent paths.
    Allows testing persistence across manager instances.

    Returns IndexManager configured with persistent storage.
    """
    vector, keyword, graph = persistent_indices_isolated
    return IndexManager(persistent_config, vector, keyword, graph)


# ============================================================================
# Hybrid Fixtures (Module-Scoped Config + Function-Scoped Indices)
# ============================================================================


@pytest.fixture(scope="module")
def persistent_config_module(tmp_path_factory) -> Config:
    """
    Create module-scoped configuration with dedicated module storage.

    Each test module gets its own persistent storage directory that
    survives across tests in that module.

    Returns Config object with module-specific persistent paths.
    """
    base_path = tmp_path_factory.mktemp("module_persistent")
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
    )


@pytest.fixture
def persistent_manager_with_module_config(
    persistent_config_module: Config,
    persistent_indices_isolated: tuple[VectorIndex, KeywordIndex, GraphStore],
) -> IndexManager:
    """
    Create function-scoped manager with module-persistent paths.

    Fresh manager for each test but shares module-level storage paths.
    Balances isolation with realistic persistence testing.

    Returns IndexManager with module-scoped persistent storage.
    """
    vector, keyword, graph = persistent_indices_isolated
    return IndexManager(persistent_config_module, vector, keyword, graph)


# ============================================================================
# Cleanup Utilities
# ============================================================================


@pytest.fixture
def cleanup_persistent_indices(persistent_index_path: Path) -> Generator[None, None, None]:
    """
    Clean up persistent indices after test execution.

    Use this fixture when you need guaranteed cleanup of persistent
    storage after a test, even if using session-scoped paths.

    Example:
        def test_with_cleanup(
            persistent_manager_isolated,
            cleanup_persistent_indices
        ):
            # Test code here
            # Indices will be cleaned up after test
            pass
    """
    yield
    # Cleanup after test
    if persistent_index_path.exists():
        import shutil
        for item in persistent_index_path.iterdir():
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()


@pytest.fixture
def cleanup_persistent_docs(persistent_docs_path: Path) -> Generator[None, None, None]:
    """
    Clean up persistent documents after test execution.

    Use this fixture when you need guaranteed cleanup of persistent
    documents after a test.

    Example:
        def test_with_doc_cleanup(
            persistent_docs_path,
            cleanup_persistent_docs
        ):
            # Test code here
            # Documents will be cleaned up after test
            pass
    """
    yield
    # Cleanup after test
    if persistent_docs_path.exists():
        for item in persistent_docs_path.iterdir():
            if item.is_dir():
                import shutil
                shutil.rmtree(item)
            else:
                item.unlink()
