"""Integration tests for move detection during reconciliation.

Tests that move detection is properly integrated into the reconciliation workflow
and works end-to-end with real indices and filesystem operations.
"""

import pytest
from pathlib import Path

from src.config import Config, IndexingConfig, ChunkingConfig
from src.indexing.manager import IndexManager
from src.indexing.manifest import save_manifest, IndexManifest
from src.indexing.reconciler import build_indexed_files_map
from src.indices.vector import VectorIndex
from src.indices.keyword import KeywordIndex
from src.indices.graph import GraphStore
from src.search.orchestrator import SearchOrchestrator


def _save_manifest_for_files(manager, config, files: list[Path]):
    """Helper to save manifest with indexed files."""
    docs_path = Path(config.indexing.documents_path)
    index_path = Path(config.indexing.index_path)
    manifest = IndexManifest(
        spec_version="1.0.0",
        embedding_model="local",
        parsers={},
        chunking_config={},
        indexed_files=build_indexed_files_map([str(f) for f in files], docs_path),
    )
    save_manifest(index_path, manifest)


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
    return SearchOrchestrator(vector, keyword, graph, config, manager)


@pytest.mark.asyncio
async def test_reconciliation_detects_moves(config, manager, orchestrator):
    """Test that reconciliation triggers move detection."""
    docs_path = Path(config.indexing.documents_path)
    docs_path.mkdir(parents=True, exist_ok=True)

    # Create and index a document
    original = docs_path / "original.md"
    content = "# Test Document\n\nContent for move detection testing.\n\nThis has multiple paragraphs."
    original.write_text(content)

    manager.index_document(str(original))
    manager.persist()
    _save_manifest_for_files(manager, config, [original])

    # Verify indexed
    doc_count_before = manager.get_document_count()
    assert doc_count_before >= 1

    # Move file (simulate filesystem move)
    moved = docs_path / "moved.md"
    moved.write_text(content)
    original.unlink()

    # Discover files
    discovered = [str(moved)]

    # Trigger reconciliation (this should detect move)
    result = manager.reconcile_indices(discovered, docs_path)

    # Assert: Move was detected and applied
    assert result.moved_count == 1, f"Expected 1 move, got {result.moved_count}"
    assert result.removed_count == 0, f"Should not remove when move detected, got {result.removed_count}"
    assert result.added_count == 0, f"Should not add when move detected, got {result.added_count}"

    # Verify index has been updated (persist + query would require full end-to-end test)
    # For now, just verify the move was tracked
    assert manager.get_document_count() >= 1


@pytest.mark.asyncio
async def test_reconciliation_respects_move_threshold(config, manager):
    """Test that partial edits during move fall back to reindex."""
    docs_path = Path(config.indexing.documents_path)
    docs_path.mkdir(parents=True, exist_ok=True)

    # Create and index a document with many chunks
    original = docs_path / "original.md"
    content_lines = ["# Test\n\n"]
    for i in range(20):
        content_lines.append(f"## Section {i}\n\nContent paragraph {i} with some text.\n\n")
    original.write_text("".join(content_lines))

    manager.index_document(str(original))
    manager.persist()
    _save_manifest_for_files(manager, config, [original])

    # Move file and edit 50% of content (below 80% threshold)
    moved = docs_path / "moved.md"
    modified_lines = ["# Test\n\n"]
    for i in range(20):
        if i < 10:
            # First 10 sections: modified
            modified_lines.append(f"## Modified Section {i}\n\nCOMPLETELY NEW CONTENT {i}.\n\n")
        else:
            # Last 10 sections: unchanged
            modified_lines.append(f"## Section {i}\n\nContent paragraph {i} with some text.\n\n")
    moved.write_text("".join(modified_lines))
    original.unlink()

    # Discover files
    discovered = [str(moved)]

    # Trigger reconciliation
    result = manager.reconcile_indices(discovered, docs_path)

    # Assert: Falls back to full reindex (below 80% threshold)
    assert result.moved_count == 0, f"Move should NOT be detected (below threshold), got {result.moved_count}"
    assert result.removed_count == 1, f"Should remove old doc, got {result.removed_count}"
    assert result.added_count == 1, f"Should add new doc, got {result.added_count}"


@pytest.mark.asyncio
async def test_move_detection_disabled_via_config(config, manager):
    """Test that move detection can be disabled."""
    # Disable move detection
    config.indexing.enable_move_detection = False

    docs_path = Path(config.indexing.documents_path)
    docs_path.mkdir(parents=True, exist_ok=True)

    # Create, index, and move a file
    original = docs_path / "original.md"
    content = "# Test\n\nContent for testing disabled move detection."
    original.write_text(content)
    manager.index_document(str(original))
    manager.persist()
    _save_manifest_for_files(manager, config, [original])

    moved = docs_path / "moved.md"
    moved.write_text(content)
    original.unlink()

    # Discover files
    discovered = [str(moved)]

    # Trigger reconciliation
    result = manager.reconcile_indices(discovered, docs_path)

    # Assert: Move detection not used (treated as remove + add)
    assert result.moved_count == 0, f"Move detection should be disabled, got {result.moved_count}"
    assert result.removed_count == 1, f"Should remove old doc, got {result.removed_count}"
    assert result.added_count == 1, f"Should add new doc, got {result.added_count}"


@pytest.mark.asyncio
async def test_move_with_partial_edit_above_threshold(config, manager, orchestrator):
    """Test that moves with minor edits (above threshold) are detected."""
    docs_path = Path(config.indexing.documents_path)
    docs_path.mkdir(parents=True, exist_ok=True)

    # Create document with 10 sections
    original = docs_path / "doc.md"
    content_lines = ["# Document\n\n"]
    for i in range(10):
        content_lines.append(f"## Section {i}\n\nParagraph {i} content.\n\n")
    original.write_text("".join(content_lines))

    manager.index_document(str(original))
    manager.persist()
    _save_manifest_for_files(manager, config, [original])

    # Move and edit only 1 section (90% match, above 80% threshold)
    moved = docs_path / "renamed.md"
    modified_lines = ["# Document\n\n"]
    for i in range(10):
        if i == 0:
            # Edit first section only
            modified_lines.append(f"## Updated Section {i}\n\nNEW CONTENT HERE.\n\n")
        else:
            # Keep other sections unchanged
            modified_lines.append(f"## Section {i}\n\nParagraph {i} content.\n\n")
    moved.write_text("".join(modified_lines))
    original.unlink()

    # Discover files
    discovered = [str(moved)]

    # Trigger reconciliation
    result = manager.reconcile_indices(discovered, docs_path)

    # Assert: Move detected (90% >= 80% threshold)
    assert result.moved_count == 1, f"Expected move detection (90% match), got moved={result.moved_count}"
    assert result.removed_count == 0
    assert result.added_count == 0


@pytest.mark.asyncio
async def test_multiple_moves_in_one_reconciliation(config, manager):
    """Test detecting multiple file moves in a single reconciliation."""
    docs_path = Path(config.indexing.documents_path)
    docs_path.mkdir(parents=True, exist_ok=True)

    # Create and index 3 files
    files = []
    for i in range(3):
        file_path = docs_path / f"file{i}.md"
        file_path.write_text(f"# File {i}\n\nContent for file number {i}.")
        manager.index_document(str(file_path))
        files.append(file_path)

    manager.persist()
    _save_manifest_for_files(manager, config, files)

    # Move all 3 files
    moved_files = []
    for i, original_path in enumerate(files):
        moved_path = docs_path / f"renamed{i}.md"
        moved_path.write_text(f"# File {i}\n\nContent for file number {i}.")
        original_path.unlink()
        moved_files.append(moved_path)

    # Discover moved files
    discovered = [str(f) for f in moved_files]

    # Trigger reconciliation
    result = manager.reconcile_indices(discovered, docs_path)

    # Assert: All 3 moves detected
    assert result.moved_count == 3, f"Expected 3 moves, got {result.moved_count}"
    assert result.removed_count == 0
    assert result.added_count == 0


@pytest.mark.asyncio
async def test_move_with_subdirectory(config, manager, orchestrator):
    """Test move detection works with files in subdirectories."""
    docs_path = Path(config.indexing.documents_path)
    docs_path.mkdir(parents=True, exist_ok=True)

    # Create file in root
    original = docs_path / "root_doc.md"
    content = "# Root Document\n\nThis is a root-level document."
    original.write_text(content)
    manager.index_document(str(original))
    manager.persist()
    _save_manifest_for_files(manager, config, [original])

    # Move to subdirectory
    subdir = docs_path / "archive"
    subdir.mkdir()
    moved = subdir / "root_doc.md"
    moved.write_text(content)
    original.unlink()

    # Discover files
    discovered = [str(moved)]

    # Trigger reconciliation
    result = manager.reconcile_indices(discovered, docs_path)

    # Assert: Move detected across directory levels
    assert result.moved_count == 1
    assert result.removed_count == 0
    assert result.added_count == 0


@pytest.mark.asyncio
async def test_move_detection_with_failed_move(config, manager):
    """Test that failed moves fall back to reindex."""
    docs_path = Path(config.indexing.documents_path)
    docs_path.mkdir(parents=True, exist_ok=True)

    # Create and index file
    original = docs_path / "test.md"
    original.write_text("# Test\n\nOriginal content.")
    manager.index_document(str(original))
    manager.persist()
    _save_manifest_for_files(manager, config, [original])

    # Corrupt vector index to force move failure
    manager.vector._index = None

    # Move file
    moved = docs_path / "moved.md"
    moved.write_text("# Test\n\nOriginal content.")
    original.unlink()

    # Discover files
    discovered = [str(moved)]

    # Trigger reconciliation (should not crash)
    result = manager.reconcile_indices(discovered, docs_path)

    # Should fall back to full reindex
    # Move would fail, so it should be treated as remove + add
    assert result.moved_count == 0 or result.failed_count > 0
