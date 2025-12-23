"""
Integration tests demonstrating persistent database fixtures.

These tests validate the persistent fixture infrastructure and serve
as examples for using different fixture scopes.
"""

from pathlib import Path

import pytest

from src.indexing.manifest import IndexManifest, load_manifest, save_manifest
from src.indices.graph import GraphStore
from src.indices.keyword import KeywordIndex
from src.indices.vector import VectorIndex


def test_persistent_config_has_correct_paths(persistent_config, persistent_docs_path, persistent_index_path):
    """
    Verify persistent config uses correct persistent paths.

    Validates that session-scoped configuration points to persistent storage.
    """
    assert persistent_config.indexing.documents_path == str(persistent_docs_path)
    assert persistent_config.indexing.index_path == str(persistent_index_path)
    assert persistent_docs_path.exists()
    assert persistent_index_path.exists()


def test_persistent_manager_can_persist_and_load(
    persistent_manager_isolated,
    persistent_config,
    persistent_docs_path,
):
    """
    Test manager can persist indices and load them back.

    Validates basic persistence cycle: index → persist → load → verify.
    Uses function-scoped manager for isolation.
    """
    # Create test document
    test_file = persistent_docs_path / "test_persist.md"
    test_file.write_text("# Test Persistence\n\nThis tests persistence behavior.")

    # Index document
    manager1 = persistent_manager_isolated
    manager1.index_document(str(test_file))

    # Verify document is indexed
    doc_count1 = manager1.get_document_count()
    assert doc_count1 > 0

    # Persist indices
    manager1.persist()

    # Create new manager instance and load
    vector2 = VectorIndex()
    keyword2 = KeywordIndex()
    graph2 = GraphStore()
    from src.indexing.manager import IndexManager
    manager2 = IndexManager(persistent_config, vector2, keyword2, graph2)
    manager2.load()

    # Verify loaded data
    doc_count2 = manager2.get_document_count()
    assert doc_count2 > 0


def test_persistent_paths_survive_across_function_calls(persistent_index_path):
    """
    Verify persistent paths maintain data across function calls.

    Tests that session-scoped paths persist data across multiple test functions.
    """
    # Create marker file in persistent storage
    marker_file = persistent_index_path / "test_marker.txt"
    marker_file.write_text("This file should persist")

    # Verify file exists (will be checked in next test)
    assert marker_file.exists()


def test_persistent_paths_still_have_data(persistent_index_path):
    """
    Verify data from previous test still exists.

    Demonstrates session-scoped persistence across test functions.
    Note: This test depends on test execution order, which is typically
    not recommended, but is used here to demonstrate persistence behavior.
    """
    marker_file = persistent_index_path / "test_marker.txt"

    # File should still exist from previous test
    assert marker_file.exists()
    assert marker_file.read_text() == "This file should persist"


def test_manifest_persistence_with_persistent_fixtures(
    persistent_config,
    persistent_index_path,
):
    """
    Test manifest save and load with persistent storage.

    Validates manifest persistence behavior using persistent paths.
    """
    # Create and save manifest
    manifest = IndexManifest(
        spec_version="1.0.0",
        embedding_model="test-model",
        parsers={"**/*.md": "MarkdownParser"},
    )
    save_manifest(persistent_index_path, manifest)

    # Verify manifest file exists
    manifest_file = persistent_index_path / "index.manifest.json"
    assert manifest_file.exists()

    # Load and verify manifest
    loaded_manifest = load_manifest(persistent_index_path)
    assert loaded_manifest is not None
    assert loaded_manifest.spec_version == "1.0.0"
    assert loaded_manifest.embedding_model == "test-model"


def test_cleanup_fixture_removes_indices(
    persistent_manager_isolated,
    persistent_index_path,
    cleanup_persistent_indices,
):
    """
    Test cleanup fixture removes indices after test.

    Demonstrates cleanup fixture usage for guaranteed cleanup.
    """
    # Create some index data
    manager = persistent_manager_isolated
    manager.persist()  # Creates index directories

    # Note: Cleanup happens after test completes
    # Verification of cleanup happens in next test


def test_after_cleanup_indices_are_gone(persistent_index_path):
    """
    Verify cleanup fixture removed indices from previous test.

    Validates that cleanup_persistent_indices fixture worked.
    Note: This depends on test execution order for demonstration.
    """
    # After cleanup, index directories should not exist or be empty
    # (session-scoped path still exists, but should be clean)
    vector_path = persistent_index_path / "vector"

    # Directory might exist but should be empty or non-existent
    if vector_path.exists():
        # If it exists, it should be empty or only have docstore.json
        files = list(vector_path.iterdir())
        # Either empty or just infrastructure files
        assert len(files) <= 1


def test_module_config_provides_isolated_paths(persistent_config_module):
    """
    Test module-scoped config provides isolated storage.

    Validates that module-scoped fixtures get their own storage paths
    separate from session-scoped storage.
    """
    # Module config should have its own paths
    docs_path = Path(persistent_config_module.indexing.documents_path)
    index_path = Path(persistent_config_module.indexing.index_path)

    assert docs_path.exists()
    assert index_path.exists()
    assert "module_persistent" in str(index_path)


@pytest.mark.asyncio
async def test_persistent_fixtures_work_with_async_tests(persistent_manager_isolated, persistent_docs_path):
    """
    Test persistent fixtures work in async test context.

    Validates that fixtures are compatible with async tests.
    """
    # Create test document
    test_file = persistent_docs_path / "async_test.md"
    test_file.write_text("# Async Test\n\nTesting async compatibility.")

    # Index document
    manager = persistent_manager_isolated
    manager.index_document(str(test_file))

    # Verify indexing worked
    doc_count = manager.get_document_count()
    assert doc_count > 0


def test_ephemeral_vs_persistent_comparison(tmp_path, persistent_docs_path):
    """
    Demonstrate difference between ephemeral and persistent storage.

    Shows the different use cases for tmp_path vs persistent fixtures.
    """
    # Ephemeral: Fast, isolated, discarded after test
    ephemeral_file = tmp_path / "ephemeral.md"
    ephemeral_file.write_text("Ephemeral data")
    assert ephemeral_file.exists()
    # This file will be gone after test completes

    # Persistent: Survives test completion, shared across session
    persistent_file = persistent_docs_path / "persistent.md"
    persistent_file.write_text("Persistent data")
    assert persistent_file.exists()
    # This file survives until session cleanup


def test_persistent_file_from_previous_test_still_exists(persistent_docs_path):
    """
    Verify persistent file from previous test survived.

    Demonstrates session-level persistence.
    """
    persistent_file = persistent_docs_path / "persistent.md"

    # File should still exist from previous test
    assert persistent_file.exists()
    assert persistent_file.read_text() == "Persistent data"
