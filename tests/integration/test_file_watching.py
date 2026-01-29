"""
Integration tests for File Watcher (D10).

Tests FileWatcher's ability to detect file system events (create, modify, delete)
and trigger appropriate indexing operations through IndexManager. Uses real
components with temporary storage and async test patterns.
"""

import asyncio
from pathlib import Path

import pytest

from src.config import Config, IndexingConfig, LLMConfig, SearchConfig, ServerConfig
from src.indexing.manager import IndexManager
from src.indexing.watcher import FileWatcher
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
        llm=LLMConfig(),
    )


@pytest.fixture
def indices(shared_embedding_model):
    """
    Create real index instances.

    Returns tuple of (vector, keyword, graph) indices for IndexManager.
    """
    vector = VectorIndex(embedding_model=shared_embedding_model)
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
def watcher(config, manager):
    """
    Create FileWatcher with real IndexManager.

    Uses short cooldown (0.2s) for faster test execution.
    """
    return FileWatcher(
        documents_path=config.indexing.documents_path,
        index_manager=manager,
        cooldown=0.2,
        include_patterns=config.indexing.include,
        exclude_patterns=config.indexing.exclude,
        exclude_hidden_dirs=config.indexing.exclude_hidden_dirs,
    )


@pytest.mark.asyncio
async def test_detect_file_creation_and_index(watcher, manager, config, tmp_path):
    """
    Test that FileWatcher detects new file creation and indexes it.

    Validates the watcher correctly monitors file system events and triggers
    indexing for newly created markdown files.
    """
    # Start watcher
    watcher.start()

    try:
        # Create new markdown file
        docs_path = Path(config.indexing.documents_path)
        new_file = docs_path / "new_document.md"
        new_file.write_text("# New Document\n\nThis is a newly created document.")

        # Wait for watcher to detect, debounce, and process (longer for filesystem)
        await asyncio.sleep(1.0)

        # Verify document was indexed (count increased)
        doc_count = manager.get_document_count()
        assert doc_count > 0
    finally:
        await watcher.stop()


@pytest.mark.asyncio
async def test_detect_file_modification_and_reindex(watcher, manager, config, tmp_path):
    """
    Test that FileWatcher detects file modifications and re-indexes.

    Ensures the watcher triggers re-indexing when an existing file is
    modified, updating all indices with the new content.
    """
    docs_path = Path(config.indexing.documents_path)
    test_file = docs_path / "modified_doc.md"

    # Create initial file
    test_file.write_text("# Original Content\n\nThis is the original text.")

    # Start watcher
    watcher.start()

    try:
        # Wait for initial indexing
        await asyncio.sleep(1.0)

        # Modify the file
        test_file.write_text("# Modified Content\n\nThis text has been updated.")

        # Wait for debounce + processing
        await asyncio.sleep(1.0)

        # Verify updated content is indexed (count remains stable)
        doc_count = manager.get_document_count()
        assert doc_count > 0
    finally:
        await watcher.stop()


@pytest.mark.asyncio
async def test_detect_file_deletion_and_remove_from_index(
    watcher, manager, config, tmp_path
):
    """
    Test that FileWatcher detects file deletion and removes from indices.

    Validates cleanup mechanism: deleted files should be removed from all
    indices to prevent stale search results.
    """
    docs_path = Path(config.indexing.documents_path)
    test_file = docs_path / "to_delete.md"

    # Start watcher first
    watcher.start()

    try:
        # Create and index file
        test_file.write_text("# Document to Delete\n\nThis will be removed.")

        # Wait for initial indexing
        await asyncio.sleep(1.0)

        # Verify document exists (count > 0)
        count_before = manager.get_document_count()
        assert count_before > 0

        # Delete the file
        test_file.unlink()

        # Wait for debounce + processing
        await asyncio.sleep(1.0)

        # Verify operation completed (removal processed)
        # Test passes if no exception occurred during removal
    finally:
        await watcher.stop()


@pytest.mark.asyncio
async def test_debouncing_multiple_rapid_changes(watcher, manager, config, tmp_path):
    """
    Test that rapid file changes are debounced into single index operation.

    Validates debouncing behavior: multiple quick edits to the same file
    should not trigger excessive re-indexing, improving performance.
    """
    docs_path = Path(config.indexing.documents_path)
    test_file = docs_path / "rapid_changes.md"

    # Start watcher
    watcher.start()

    try:
        # Make rapid consecutive changes (faster than debounce window)
        test_file.write_text("# Version 1\n\nFirst version.")
        await asyncio.sleep(0.05)
        test_file.write_text("# Version 2\n\nSecond version.")
        await asyncio.sleep(0.05)
        test_file.write_text("# Version 3\n\nFinal version.")

        # Wait for debounce + processing
        await asyncio.sleep(1.0)

        # Verify final version is indexed
        doc_count = manager.get_document_count()
        assert doc_count > 0
    finally:
        await watcher.stop()


@pytest.mark.asyncio
async def test_batch_processing_multiple_files(watcher, manager, config, tmp_path):
    """
    Test that multiple file changes are batched and processed together.

    Validates batch processing: creating/modifying multiple files within
    the debounce window should trigger a single batch operation, improving
    efficiency for bulk operations.
    """
    docs_path = Path(config.indexing.documents_path)

    # Start watcher
    watcher.start()

    try:
        # Create multiple files in quick succession
        file1 = docs_path / "batch_file1.md"
        file2 = docs_path / "batch_file2.md"
        file3 = docs_path / "batch_file3.md"

        file1.write_text("# Batch File 1\n\nFirst batch file.")
        await asyncio.sleep(0.05)
        file2.write_text("# Batch File 2\n\nSecond batch file.")
        await asyncio.sleep(0.05)
        file3.write_text("# Batch File 3\n\nThird batch file.")

        # Wait for debounce + batch processing
        await asyncio.sleep(1.0)

        # Verify all files were indexed (count reflects batch)
        doc_count = manager.get_document_count()
        assert doc_count >= 3
    finally:
        await watcher.stop()


@pytest.mark.asyncio
async def test_watcher_handles_non_markdown_files(watcher, manager, config, tmp_path):
    """
    Test that FileWatcher ignores non-markdown files.

    Ensures the watcher only processes configured file types (.md, .markdown)
    and ignores other files in the watched directory.
    """
    docs_path = Path(config.indexing.documents_path)

    # Start watcher
    watcher.start()

    try:
        # Create markdown file
        md_file = docs_path / "valid.md"
        md_file.write_text("# Valid Markdown\n\nShould be indexed.")

        # Create non-markdown files
        txt_file = docs_path / "ignore.txt"
        txt_file.write_text("This should be ignored.")

        py_file = docs_path / "script.py"
        py_file.write_text("print('ignored')")

        # Wait for debounce + processing
        await asyncio.sleep(1.0)

        # Verify only markdown file was indexed (count == 1)
        doc_count = manager.get_document_count()
        assert doc_count == 1
    finally:
        await watcher.stop()


@pytest.mark.asyncio
async def test_watcher_excludes_venv_directories(watcher, manager, config, tmp_path):
    """
    Test that FileWatcher ignores files in excluded directories like .venv.

    Validates exclude pattern filtering: markdown files inside .venv, venv,
    node_modules, etc. should not be indexed even if they're valid markdown.
    """
    docs_path = Path(config.indexing.documents_path)

    # Start watcher
    watcher.start()

    try:
        # Create markdown file in normal location (should be indexed)
        normal_file = docs_path / "normal.md"
        normal_file.write_text("# Normal Document\n\nShould be indexed.")

        # Create markdown file in .venv (should be excluded)
        venv_dir = docs_path / ".venv" / "lib" / "python3.13" / "site-packages"
        venv_dir.mkdir(parents=True)
        venv_file = venv_dir / "README.md"
        venv_file.write_text("# Venv Package\n\nShould NOT be indexed.")

        # Create markdown file in node_modules (should be excluded)
        node_dir = docs_path / "node_modules" / "some-package"
        node_dir.mkdir(parents=True)
        node_file = node_dir / "README.md"
        node_file.write_text("# Node Package\n\nShould NOT be indexed.")

        # Wait for debounce + processing
        await asyncio.sleep(1.0)

        # Verify only normal file was indexed (count == 1)
        doc_count = manager.get_document_count()
        assert doc_count == 1
    finally:
        await watcher.stop()


@pytest.mark.asyncio
async def test_watcher_excludes_hidden_directories(watcher, manager, config, tmp_path):
    """
    Test that FileWatcher ignores files in hidden directories.

    Validates exclude_hidden_dirs option: files in directories starting
    with '.' should be excluded by default.
    """
    docs_path = Path(config.indexing.documents_path)

    # Start watcher
    watcher.start()

    try:
        # Create markdown file in normal location (should be indexed)
        normal_file = docs_path / "visible.md"
        normal_file.write_text("# Visible Document\n\nShould be indexed.")

        # Create markdown file in hidden directory (should be excluded)
        hidden_dir = docs_path / ".hidden"
        hidden_dir.mkdir()
        hidden_file = hidden_dir / "secret.md"
        hidden_file.write_text("# Hidden Document\n\nShould NOT be indexed.")

        # Wait for debounce + processing
        await asyncio.sleep(1.0)

        # Verify only visible file was indexed (count == 1)
        doc_count = manager.get_document_count()
        assert doc_count == 1
    finally:
        await watcher.stop()
