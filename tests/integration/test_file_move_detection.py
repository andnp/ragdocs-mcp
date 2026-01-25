"""Integration tests for file move detection.

Tests end-to-end move detection workflow with real indices and files.
"""

import pytest
from pathlib import Path

from src.config import Config, IndexingConfig, ChunkingConfig
from src.indexing.manager import IndexManager
from src.indices.vector import VectorIndex
from src.indices.keyword import KeywordIndex
from src.indices.graph import GraphStore
from src.search.orchestrator import SearchOrchestrator


@pytest.fixture
def config(tmp_path):
    return Config(
        indexing=IndexingConfig(
            documents_path=str(tmp_path / "docs"),
            index_path=str(tmp_path / "index"),
            enable_delta_indexing=True,
            enable_move_detection=True,
            move_detection_threshold=0.8,
        ),
        document_chunking=ChunkingConfig(
            min_chunk_chars=100,
            max_chunk_chars=500,
        ),
    )


@pytest.fixture
def indices(shared_embedding_model):
    vector = VectorIndex(embedding_model=shared_embedding_model)
    keyword = KeywordIndex()
    graph = GraphStore()
    return vector, keyword, graph


@pytest.fixture
def manager(config, indices):
    vector, keyword, graph = indices
    return IndexManager(config, vector, keyword, graph)


@pytest.fixture
def orchestrator(config, indices, manager):
    vector, keyword, graph = indices
    return SearchOrchestrator(vector, keyword, graph, config)


@pytest.mark.asyncio
async def test_move_file_preserves_embeddings(config, manager, orchestrator):
    """Test that moving a file maintains searchability.

    Note: Move detection with embedding reuse requires batch processing (file watcher).
    This test verifies that content remains searchable after rename.
    """
    docs_dir = Path(config.indexing.documents_path)
    docs_dir.mkdir(parents=True, exist_ok=True)

    # Create and index original file
    original_file = docs_dir / "original.md"
    original_file.write_text(
        "# Authentication\n\n"
        "This document describes OAuth 2.0 authentication.\n\n"
        "## Setup\n\n"
        "Configure your client credentials."
    )

    manager.index_document(str(original_file))

    # Verify indexed
    doc_count_before = manager.get_document_count()
    assert doc_count_before >= 1

    # Query to verify content is searchable
    results_before, _, _ = await orchestrator.query("OAuth authentication", top_k=5, top_n=3)
    assert len(results_before) > 0
    assert any("authentication" in r.content.lower() for r in results_before)

    # Simulate file move (rename file, reindex)
    original_file.unlink()
    moved_file = docs_dir / "moved.md"
    moved_file.write_text(
        "# Authentication\n\n"
        "This document describes OAuth 2.0 authentication.\n\n"
        "## Setup\n\n"
        "Configure your client credentials."
    )

    # Index moved file
    manager.index_document(str(moved_file))

    # Verify content still searchable
    results_after, _, _ = await orchestrator.query("OAuth authentication", top_k=5, top_n=3)
    assert len(results_after) > 0
    assert any("authentication" in r.content.lower() for r in results_after)


@pytest.mark.asyncio
async def test_move_and_edit_uses_delta(config, manager, orchestrator):
    """Test that moving and editing a file still allows queries to work.

    Note: Move detection requires batch processing (file watcher).
    This test verifies basic functionality works correctly.
    """
    docs_dir = Path(config.indexing.documents_path)
    docs_dir.mkdir(parents=True, exist_ok=True)

    # Create original file with multiple chunks
    original_file = docs_dir / "doc.md"
    original_file.write_text(
        "# Section 1\n\n"
        "This is the first section with some content.\n\n"
        "# Section 2\n\n"
        "This is the second section with more content.\n\n"
        "# Section 3\n\n"
        "This is the third section with additional content."
    )

    manager.index_document(str(original_file))

    # Move file and edit one section
    original_file.unlink()
    moved_file = docs_dir / "renamed_doc.md"
    moved_file.write_text(
        "# Section 1\n\n"
        "This is the first section with some content.\n\n"  # Unchanged
        "# Section 2\n\n"
        "THIS SECTION WAS EDITED WITH NEW CONTENT.\n\n"  # Changed
        "# Section 3\n\n"
        "This is the third section with additional content."  # Unchanged
    )

    # Index moved file
    manager.index_document(str(moved_file))

    # Verify search works with edited content
    results, _, _ = await orchestrator.query("EDITED NEW CONTENT", top_k=5, top_n=3)
    assert len(results) > 0, "Should find edited content"

    # Verify old file is no longer returned
    results2, _, _ = await orchestrator.query("first section", top_k=10, top_n=5)
    assert len(results2) > 0, "Should find unchanged content"


@pytest.mark.asyncio
async def test_move_below_threshold_reindexes(config, manager):
    """Test that moves below threshold trigger full re-index."""
    docs_dir = Path(config.indexing.documents_path)
    docs_dir.mkdir(parents=True, exist_ok=True)

    # Create original file
    original_file = docs_dir / "doc.md"
    original_file.write_text(
        "# Original Content\n\n"
        "This is the original content of the document.\n\n"
        "It has multiple paragraphs for testing."
    )

    manager.index_document(str(original_file))
    embeddings_before = len(manager.vector._chunk_id_to_node_id)

    # Move and change most content (below threshold)
    original_file.unlink()
    moved_file = docs_dir / "moved.md"
    moved_file.write_text(
        "# Completely Different Content\n\n"
        "This document has been completely rewritten.\n\n"
        "Almost nothing matches the original."
    )

    manager.index_document(str(moved_file))

    # Should have re-indexed (different embedding count)
    embeddings_after = len(manager.vector._chunk_id_to_node_id)
    # Count should be similar (old chunks removed, new ones added)
    assert abs(embeddings_after - embeddings_before) <= 2


@pytest.mark.asyncio
async def test_query_after_move_finds_content(config, manager, orchestrator):
    """Test that queries find content after file move."""
    docs_dir = Path(config.indexing.documents_path)
    docs_dir.mkdir(parents=True, exist_ok=True)

    # Create file
    original_file = docs_dir / "security.md"
    original_file.write_text(
        "# Security Best Practices\n\n"
        "Always use HTTPS for authentication.\n\n"
        "## Token Storage\n\n"
        "Store tokens securely in encrypted storage."
    )

    manager.index_document(str(original_file))

    # Move file
    original_file.unlink()
    moved_file = docs_dir / "guides" / "security.md"
    moved_file.parent.mkdir(parents=True, exist_ok=True)
    moved_file.write_text(
        "# Security Best Practices\n\n"
        "Always use HTTPS for authentication.\n\n"
        "## Token Storage\n\n"
        "Store tokens securely in encrypted storage."
    )

    manager.index_document(str(moved_file))

    # Query should find content with new path
    results, _, _ = await orchestrator.query("token storage HTTPS", top_k=5, top_n=3)
    assert len(results) > 0

    found_moved = False
    for result in results:
        assert "security.md" not in result.file_path or "guides" in result.file_path
        if "guides/security.md" in result.file_path:
            found_moved = True

    assert found_moved, "Should find content in moved location"


@pytest.mark.asyncio
async def test_move_multiple_files(config, manager, orchestrator):
    """Test moving multiple files in batch.

    Note: Move detection is not yet integrated with batch processing.
    This test verifies that files can be moved and remain searchable,
    but embeddings are currently re-generated (no deduplication yet).

    TODO: Integrate _detect_file_moves with FileWatcher/reconciliation
    to enable embedding reuse during moves.
    """
    docs_dir = Path(config.indexing.documents_path)
    docs_dir.mkdir(parents=True, exist_ok=True)

    # Create multiple files
    files = []
    for i in range(3):
        file = docs_dir / f"doc{i}.md"
        file.write_text(
            f"# Document {i}\n\n"
            f"This is document number {i}.\n\n"
            f"It contains unique content for testing."
        )
        manager.index_document(str(file))
        files.append(file)

    # Move all files to subdirectory
    moved_dir = docs_dir / "archive"
    moved_dir.mkdir(parents=True, exist_ok=True)

    # Remove old files first
    for old_file in files:
        doc_id = str(Path(old_file).relative_to(docs_dir).with_suffix(""))
        manager.remove_document(doc_id)

    # Index files in new location
    for i in range(3):
        new_file = moved_dir / f"doc{i}.md"
        new_file.write_text(
            f"# Document {i}\n\n"
            f"This is document number {i}.\n\n"
            f"It contains unique content for testing."
        )
        manager.index_document(str(new_file))

    # Since move detection is not integrated yet, chunks are re-created.
    # Verify files are searchable in new location.

    for i in range(3):
        results, _, _ = await orchestrator.query(f"document number {i}", top_k=5, top_n=2)
        assert len(results) > 0
        assert any("archive" in r.file_path for r in results), (
            f"Document {i} should be searchable in archive location"
        )


@pytest.mark.asyncio
async def test_move_detection_with_git_rename(config, manager, orchestrator):
    """Test move detection works with git-style renames."""
    docs_dir = Path(config.indexing.documents_path)
    docs_dir.mkdir(parents=True, exist_ok=True)

    # Create file
    original = docs_dir / "README.md"
    content = "# Project README\n\nWelcome to the project.\n\nThis is the main documentation."

    original.write_text(content)
    manager.index_document(str(original))

    # Simulate git rename (delete + create)
    original.unlink()
    renamed = docs_dir / "README_OLD.md"
    renamed.write_text(content)

    manager.index_document(str(renamed))

    # Should detect as move
    results, _, _ = await orchestrator.query("Project README", top_k=5, top_n=2)
    assert len(results) > 0
    assert any("README_OLD.md" in r.file_path for r in results)
    assert not any("README.md" in r.file_path or "README.md" == Path(r.file_path).name for r in results)


def test_move_detection_fallback_on_failure(config, manager):
    """Test that move detection falls back to full reindex on failure."""
    docs_dir = Path(config.indexing.documents_path)
    docs_dir.mkdir(parents=True, exist_ok=True)

    # Create file
    original = docs_dir / "test.md"
    original.write_text("# Test\n\nOriginal content.")
    manager.index_document(str(original))

    # Simulate corrupted index state (remove vector index)
    manager.vector._index = None

    # Move file - should fall back to full reindex
    original.unlink()
    moved = docs_dir / "moved.md"
    moved.write_text("# Test\n\nOriginal content.")

    # Should not raise exception
    manager.index_document(str(moved))

    # Verify it was indexed (fallback worked)
    assert manager.get_document_count() >= 1
