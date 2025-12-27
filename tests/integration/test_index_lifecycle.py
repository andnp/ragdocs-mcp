"""
Integration tests for Index Lifecycle Management (D11).

Tests the IndexManager's ability to detect and respond to manifest changes,
triggering full rebuilds when necessary and skipping rebuilds when the
manifest matches the current configuration.
"""

import glob
from pathlib import Path

import pytest

from src.config import Config, IndexingConfig, LLMConfig, SearchConfig, ServerConfig
from src.indexing.manager import IndexManager
from src.indexing.manifest import IndexManifest, load_manifest, save_manifest, should_rebuild
from src.indices.graph import GraphStore
from src.indices.keyword import KeywordIndex
from src.indices.vector import VectorIndex


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
        search=SearchConfig(),
        llm=LLMConfig(embedding_model="all-MiniLM-L6-v2"),
    )


@pytest.fixture
def indices():
    """
    Create real index instances.

    Returns tuple of (vector, keyword, graph) indices for IndexManager.
    """
    vector = VectorIndex()
    keyword = KeywordIndex()
    graph = GraphStore()
    return vector, keyword, graph


@pytest.fixture
def manager(config, indices):
    """
    Create IndexManager with real indices.

    Provides fully functional manager for integration testing.
    """
    vector, keyword, graph = indices
    return IndexManager(config, vector, keyword, graph)


@pytest.fixture
def current_manifest(config):
    """
    Generate current manifest from configuration.

    Represents the manifest that would be generated during startup.
    """
    return IndexManifest(
        spec_version="1.0.0",
        embedding_model=config.llm.embedding_model,
        parsers=config.parsers,
        chunking_config={},
    )


def index_all_documents(manager, docs_path):
    """
    Index all markdown files in the documents directory.

    Helper function to simulate full indexing on startup.
    """
    pattern = str(Path(docs_path) / "**" / "*.md")
    for file_path in glob.glob(pattern, recursive=True):
        manager.index_document(file_path)


def test_startup_no_manifest_triggers_full_index(config, manager, current_manifest, tmp_path):
    """
    Test that startup with no manifest triggers a full index.

    When no manifest exists, the system should perform a complete indexing
    of all documents and save a new manifest for future comparisons.
    """
    docs_path = Path(config.indexing.documents_path)
    index_path = Path(config.indexing.index_path)

    # Create test documents
    (docs_path / "doc1.md").write_text("# Document 1\n\nFirst document content.")
    (docs_path / "doc2.md").write_text("# Document 2\n\nSecond document content.")
    (docs_path / "doc3.md").write_text("# Document 3\n\nThird document content.")

    # Verify no manifest exists
    saved_manifest = load_manifest(index_path)
    assert saved_manifest is None

    # Check if rebuild is needed (should be True)
    needs_rebuild = should_rebuild(current_manifest, saved_manifest)
    assert needs_rebuild is True

    # Perform full indexing
    index_all_documents(manager, docs_path)

    # Save manifest after indexing
    manager.persist()
    save_manifest(index_path, current_manifest)

    # Verify all documents were indexed
    doc_count = manager.get_document_count()
    assert doc_count == 3

    # Verify manifest was saved
    saved_manifest = load_manifest(index_path)
    assert saved_manifest is not None
    assert saved_manifest.spec_version == "1.0.0"
    assert saved_manifest.embedding_model == "all-MiniLM-L6-v2"
    assert saved_manifest.parsers == {"**/*.md": "MarkdownParser"}


def test_startup_matching_manifest_skips_rebuild(config, manager, current_manifest, tmp_path):
    """
    Test that startup with matching manifest skips rebuild.

    When the saved manifest matches the current configuration, the system
    should load existing indices without performing a full reindex, saving
    time and resources.
    """
    docs_path = Path(config.indexing.documents_path)
    index_path = Path(config.indexing.index_path)

    # Create test documents
    (docs_path / "existing.md").write_text("# Existing Document\n\nPre-indexed content.")

    # Perform initial indexing
    index_all_documents(manager, docs_path)
    manager.persist()
    save_manifest(index_path, current_manifest)

    # Verify manifest exists and matches
    saved_manifest = load_manifest(index_path)
    assert saved_manifest is not None
    needs_rebuild = should_rebuild(current_manifest, saved_manifest)
    assert needs_rebuild is False

    # Simulate restart: create new manager with fresh indices
    vector_new = VectorIndex()
    keyword_new = KeywordIndex()
    graph_new = GraphStore()
    manager_new = IndexManager(config, vector_new, keyword_new, graph_new)

    # Load existing indices (skipping rebuild)
    manager_new.load()

    # Verify data was loaded correctly
    doc_count = manager_new.get_document_count()
    assert doc_count == 1  # Only the pre-indexed document


def test_startup_version_mismatch_triggers_rebuild(config, manager, current_manifest, tmp_path):
    """
    Test that startup with version mismatch triggers rebuild.

    When the spec version, embedding model, or parser configuration changes,
    the system should detect the mismatch and perform a full reindex to ensure
    all documents are processed with the updated configuration.
    """
    docs_path = Path(config.indexing.documents_path)
    index_path = Path(config.indexing.index_path)

    # Create test documents
    (docs_path / "outdated.md").write_text("# Outdated Document\n\nOld content.")

    # Perform initial indexing with old manifest
    index_all_documents(manager, docs_path)
    manager.persist()

    # Save old manifest with different version
    old_manifest = IndexManifest(
        spec_version="0.9.0",  # Old version
        embedding_model="all-MiniLM-L6-v2",
        parsers={"**/*.md": "MarkdownParser"},
        chunking_config={},
    )
    save_manifest(index_path, old_manifest)

    # Verify manifest exists but differs
    saved_manifest = load_manifest(index_path)
    assert saved_manifest is not None
    needs_rebuild = should_rebuild(current_manifest, saved_manifest)
    assert needs_rebuild is True

    # Simulate restart with version mismatch detection
    # In production, this would trigger a full reindex
    # Here we verify the detection logic works correctly

    # Add new document to verify full rebuild would capture it
    (docs_path / "new.md").write_text("# New Document\n\nNew content after upgrade.")

    # Perform full reindex
    vector_new = VectorIndex()
    keyword_new = KeywordIndex()
    graph_new = GraphStore()
    manager_new = IndexManager(config, vector_new, keyword_new, graph_new)
    index_all_documents(manager_new, docs_path)
    manager_new.persist()

    # Save updated manifest
    save_manifest(index_path, current_manifest)

    # Verify both old and new documents are indexed
    doc_count = manager_new.get_document_count()
    assert doc_count == 2  # Both outdated and new documents

    # Verify manifest was updated
    updated_manifest = load_manifest(index_path)
    assert updated_manifest is not None
    assert updated_manifest.spec_version == "1.0.0"
