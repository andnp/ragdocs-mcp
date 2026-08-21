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

# MUST be set before any HuggingFace/sentence-transformers imports to suppress
# progress bars that would pollute JSON output in E2E tests.
import contextlib
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TQDM_DISABLE"] = "1"

# The test suite must never touch the HuggingFace Hub network: real-model tests
# rely entirely on a pre-populated local cache (see scripts/download_test_models.py).
# This avoids Hub rate limiting when multiple pytest-xdist workers start at once.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# llama_index's HuggingFaceEmbedding caches to its own directory (not HF_HOME),
# and HF_HOME itself governs where transformers/sentence-transformers/
# huggingface_hub look up cached models (embedding, reranker, and the
# query-pipeline cross-encoder reranker). Both are resolved from the real HOME
# at process startup - before isolate_xdg_data_home (below) starts giving each
# test an isolated fake HOME. Pin both once, globally, here, rather than
# per-test in isolate_xdg_data_home: relying on that fixture re-deriving the
# "original" value from the CURRENT os.environ["HOME"] each test is fragile to
# fixture-ordering races (a session-scoped fixture can build a real subprocess
# before or after HOME gets faked for a given test, depending on execution
# order) - pinning here is immune to that entirely, since it never changes.
os.environ.setdefault(
    "LLAMA_INDEX_CACHE_DIR", os.path.join(os.path.expanduser("~"), ".cache", "llama_index")
)
os.environ.setdefault("HF_HOME", os.path.join(os.path.expanduser("~"), ".cache", "huggingface"))

from collections.abc import Generator
from pathlib import Path
from typing import Any

import pytest
from searchkernel.api import build_local_record_kernel
from searchkernel.domain import Vector
from searchkernel.embeddings import (
    TEST_FAKE_EMBEDDINGS_ENV_VAR,
    DeterministicFakeEmbeddingModel,
)
from searchkernel.local import LocalRecordKernel

from mcp_markdown_ragdocs.config import (
    ChunkingConfig,
    Config,
    IndexingConfig,
    LLMConfig,
    SearchConfig,
)
from mcp_markdown_ragdocs.daemon.management import inspect_daemon, stop_daemon
from mcp_markdown_ragdocs.daemon.paths import RuntimePaths
from mcp_markdown_ragdocs.indexing.record_manager import RecordIndexManager


@pytest.fixture(autouse=True)
def isolate_xdg_data_home(tmp_path_factory):
    """Isolate application data while preserving HuggingFace model cache.

    Creates temp directories for XDG_DATA_HOME and HOME to isolate test data.

    (HF_HOME and LLAMA_INDEX_CACHE_DIR are handled separately: both are pinned
    once, globally, at conftest import time - see top of this file - since
    they must be fixed before any per-test HOME isolation to be robust to
    fixture ordering. Re-deriving "the original HF_HOME" here, per test, from
    whatever os.environ["HOME"] happens to be at that moment was fragile -
    subprocess-spawning fixtures can run before or after this fixture's own
    HOME override depending on execution order.)
    """
    # Create isolated temp directories for application data
    data_home = tmp_path_factory.mktemp("xdg-data-home")
    home_dir = tmp_path_factory.mktemp("home")

    environment = pytest.MonkeyPatch()
    environment.setenv("XDG_DATA_HOME", str(data_home))
    environment.setenv("HOME", str(home_dir))

    try:
        yield
    finally:
        # Production daemons intentionally detach. Stop any daemon created
        # inside this test's isolated HOME before restoring the environment,
        # or user systemd will adopt both the daemon and its worker.
        runtime_paths = RuntimePaths.resolve()
        runtime_root = getattr(runtime_paths, "root", None)
        if runtime_root is not None and runtime_root.exists():
            with contextlib.suppress(Exception):
                metadata = inspect_daemon(runtime_paths).metadata
                # In-process lifecycle tests register the runner's own PID as
                # the daemon; stopping that would signal (and kill) the current
                # pytest-xdist worker. Only stop genuinely detached daemons.
                if metadata is None or metadata.pid != os.getpid():
                    stop_daemon(paths=runtime_paths)
        environment.undo()


# ============================================================================
# Test Fixture Factories
# ============================================================================


def make_test_config(tmp_path: Path, **overrides):
    docs_path = tmp_path / "docs"
    docs_path.mkdir(exist_ok=True)
    index_path = tmp_path / "index"
    index_path.mkdir(exist_ok=True)

    defaults: dict[str, Any] = {
        "indexing": IndexingConfig(
            documents_path=str(docs_path),
            index_path=str(index_path),
        ),
        "search": SearchConfig(),
        "chunking": ChunkingConfig(),
    }
    defaults.update(overrides)
    return Config(**defaults)


def create_test_document(docs_dir: Path | str, doc_id: str, content: str):
    doc_path = Path(docs_dir) / f"{doc_id}.md"
    doc_path.write_text(content)
    return str(doc_path)


# ============================================================================
# Fake Embedding Model Fixture
# ============================================================================


@pytest.fixture(scope="session")
def deterministic_fake_embedding_model() -> DeterministicFakeEmbeddingModel:
    """Session-scoped fake embedding model for deterministic, offline tests."""
    return DeterministicFakeEmbeddingModel()


@pytest.fixture(autouse=True)
def configure_embedding_mode_for_test(request, monkeypatch):
    """Force every test to use the deterministic, offline embedding path."""
    del request
    monkeypatch.setenv(TEST_FAKE_EMBEDDINGS_ENV_VAR, "1")


# ============================================================================
# Shared Embedding Model Fixture
# ============================================================================


@pytest.fixture(scope="session")
def shared_embedding_model():
    """Compatibility fixture returning the deterministic model used by tests."""
    return DeterministicFakeEmbeddingModel()


class DeterministicEmbeddingProvider:
    """Canonical embedding-provider adapter for offline pytest runs."""

    model_name = "__deterministic_fake__"
    dim = 384

    def __init__(self) -> None:
        self._model = DeterministicFakeEmbeddingModel(self.dim)

    def embed(self, texts: list[str]) -> list[Vector]:
        return [self._model.get_text_embedding(text) for text in texts]

    def embed_query(self, query: str) -> Vector:
        return self._model.get_query_embedding(query)


@pytest.fixture(scope="session")
def deterministic_embedding_provider() -> DeterministicEmbeddingProvider:
    """Session-scoped canonical provider with stable, network-free vectors."""
    return DeterministicEmbeddingProvider()


@pytest.fixture
def local_record_kernel(
    tmp_path: Path,
    deterministic_embedding_provider: DeterministicEmbeddingProvider,
) -> LocalRecordKernel:
    """Fresh canonical local record kernel for a unit test."""
    return build_local_record_kernel(
        tmp_path / "records.db",
        embedding_provider=deterministic_embedding_provider,
        embedding_model_name=deterministic_embedding_provider.model_name,
        embedding_dim=deterministic_embedding_provider.dim,
        vector_engine="exact",
    )


@pytest.fixture
def record_manager(
    tmp_path: Path,
    local_record_kernel: LocalRecordKernel,
    deterministic_embedding_provider: DeterministicEmbeddingProvider,
) -> RecordIndexManager:
    """Fresh app-owned manager connected to the canonical local kernel."""
    config = make_test_config(tmp_path)
    return RecordIndexManager(config, local_record_kernel, deterministic_embedding_provider)


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
        indexing=IndexingConfig(
            documents_path=str(persistent_docs_path),
            index_path=str(persistent_index_path),
        ),
        search=SearchConfig(
            semantic_weight=1.0,
            keyword_weight=1.0,
            recency_bias=0.5,
        ),
        llm=LLMConfig(embedding_model="__deterministic_fake__"),
    )


# ============================================================================
# Module-Scoped Persistent Fixtures
# ============================================================================


@pytest.fixture(scope="module")
def persistent_indices_module(
    persistent_config: Config,
    deterministic_embedding_provider: DeterministicEmbeddingProvider,
) -> Generator[LocalRecordKernel]:
    """Module-scoped canonical kernel backed by persistent SQLite records."""
    yield build_local_record_kernel(
        Path(persistent_config.indexing.index_path) / "index.db",
        embedding_provider=deterministic_embedding_provider,
        embedding_model_name=deterministic_embedding_provider.model_name,
        embedding_dim=deterministic_embedding_provider.dim,
        vector_engine="exact",
    )


@pytest.fixture(scope="module")
def persistent_manager_module(
    persistent_config: Config,
    persistent_indices_module: LocalRecordKernel,
    deterministic_embedding_provider: DeterministicEmbeddingProvider,
) -> RecordIndexManager:
    return RecordIndexManager(
        persistent_config,
        persistent_indices_module,
        deterministic_embedding_provider,
    )


# ============================================================================
# Function-Scoped Persistent Fixtures with Cleanup
# ============================================================================


@pytest.fixture
def persistent_indices_isolated(
    persistent_config: Config,
    deterministic_embedding_provider: DeterministicEmbeddingProvider,
) -> Generator[LocalRecordKernel]:
    """Fresh canonical kernel using the configured persistent database path."""
    yield build_local_record_kernel(
        Path(persistent_config.indexing.index_path) / "index.db",
        embedding_provider=deterministic_embedding_provider,
        embedding_model_name=deterministic_embedding_provider.model_name,
        embedding_dim=deterministic_embedding_provider.dim,
        vector_engine="exact",
    )


@pytest.fixture
def persistent_manager_isolated(
    persistent_config: Config,
    persistent_indices_isolated: LocalRecordKernel,
    deterministic_embedding_provider: DeterministicEmbeddingProvider,
) -> RecordIndexManager:
    return RecordIndexManager(
        persistent_config,
        persistent_indices_isolated,
        deterministic_embedding_provider,
    )


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
        indexing=IndexingConfig(
            documents_path=str(docs_path),
            index_path=str(index_path),
        ),
        search=SearchConfig(
            semantic_weight=1.0,
            keyword_weight=1.0,
            recency_bias=0.5,
        ),
        llm=LLMConfig(embedding_model="__deterministic_fake__"),
    )


@pytest.fixture
def persistent_manager_with_module_config(
    persistent_config_module: Config,
    deterministic_embedding_provider: DeterministicEmbeddingProvider,
) -> RecordIndexManager:
    kernel = build_local_record_kernel(
        Path(persistent_config_module.indexing.index_path) / "index.db",
        embedding_provider=deterministic_embedding_provider,
        embedding_model_name=deterministic_embedding_provider.model_name,
        embedding_dim=deterministic_embedding_provider.dim,
        vector_engine="exact",
    )
    return RecordIndexManager(
        persistent_config_module,
        kernel,
        deterministic_embedding_provider,
    )


# ============================================================================
# Cleanup Utilities
# ============================================================================


@pytest.fixture
def cleanup_persistent_indices(
    persistent_index_path: Path,
) -> Generator[None]:
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
def cleanup_persistent_docs(persistent_docs_path: Path) -> Generator[None]:
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


# ============================================================================
# pytest-xdist hook to handle serial tests
# ============================================================================


def pytest_xdist_auto_num_workers(config):
    """Hook to configure pytest-xdist behavior for serial tests."""
    # Let pytest-xdist determine worker count automatically
    return


def pytest_collection_modifyitems(config, items):
    """Mark serial tests to run in the main process and tag real embedding tests."""
    for item in items:
        # Mark tests that use shared_embedding_model to use real embeddings
        if "shared_embedding_model" in item.fixturenames:
            fixture_info = getattr(item, "_fixtureinfo", None)
            fixture_defs = (
                fixture_info.name2fixturedefs.get("shared_embedding_model", [])
                if fixture_info is not None
                else []
            )
            resolved_fixture = fixture_defs[-1] if fixture_defs else None
            fixture_func = getattr(resolved_fixture, "func", None)
            if (
                fixture_func is not None
                and fixture_func.__module__ == "tests.conftest"
                and fixture_func.__name__ == "shared_embedding_model"
            ):
                item.add_marker(pytest.mark.real_embeddings)

        if "serial" in item.keywords:
            # Force serial tests to run in dist group 'serial'
            # This ensures they don't run in parallel with other tests
            item.add_marker(pytest.mark.xdist_group(name="serial"))
