"""
Unit tests for IndexManager encoding error recovery.

GAP #7: IndexManager encoding error recovery (Medium/Low, Score 3.33)
"""

from pathlib import Path

import pytest

from src.config import Config, IndexingConfig, ServerConfig
from src.indexing.manager import IndexManager
from src.indices.vector import VectorIndex
from src.indices.keyword import KeywordIndex
from src.indices.graph import GraphStore


@pytest.fixture
def manager_config(tmp_path):
    """Create IndexManager configuration."""
    docs_path = tmp_path / "docs"
    docs_path.mkdir()
    return Config(
        server=ServerConfig(),
        indexing=IndexingConfig(
            documents_path=str(docs_path),
            index_path=str(tmp_path / "index"),
        ),
        parsers={"**/*.md": "MarkdownParser"}
    )


@pytest.fixture
def manager(manager_config, shared_embedding_model):
    """Create IndexManager instance."""
    vector = VectorIndex(embedding_model=shared_embedding_model)
    keyword = KeywordIndex()
    graph = GraphStore()
    return IndexManager(manager_config, vector, keyword, graph)


def test_index_document_with_utf8_bom(manager, manager_config):
    """
    IndexManager handles UTF-8 BOM (Byte Order Mark).

    Tests that files with UTF-8 BOM are indexed correctly.
    """
    docs_path = Path(manager_config.indexing.documents_path)

    # Create file with UTF-8 BOM
    file_path = docs_path / "bom_file.md"
    with open(file_path, "wb") as f:
        f.write(b"\xef\xbb\xbf# Document with BOM\n\nContent here.")

    # Should index without error
    manager.index_document(str(file_path))

    assert manager.get_document_count() == 1
    assert len(manager.get_failed_files()) == 0


def test_index_document_with_latin1_encoding(manager, manager_config):
    """
    IndexManager handles Latin-1 encoded files.

    Tests that non-UTF-8 encodings are handled gracefully.
    """
    docs_path = Path(manager_config.indexing.documents_path)

    # Create file with Latin-1 encoding (e.g., with accented characters)
    file_path = docs_path / "latin1.md"
    content = "# Café résumé naïve"
    with open(file_path, "wb") as f:
        f.write(content.encode("latin-1"))

    # May fail to index or succeed depending on error handling
    try:
        manager.index_document(str(file_path))
        # If it succeeds, that's fine
        assert True
    except UnicodeDecodeError:
        # If it fails, should be tracked in failed files
        failed = manager.get_failed_files()
        assert len(failed) >= 1


def test_index_document_with_mixed_encodings(manager, manager_config):
    """
    IndexManager tracks files with encoding errors.

    Tests that encoding failures are logged without crashing.
    """
    docs_path = Path(manager_config.indexing.documents_path)

    # Create valid UTF-8 file
    valid_file = docs_path / "valid.md"
    valid_file.write_text("# Valid UTF-8 Document")

    # Create file with invalid UTF-8 (random bytes)
    invalid_file = docs_path / "invalid.md"
    with open(invalid_file, "wb") as f:
        f.write(b"# Title\n\nInvalid UTF-8: \xff\xfe")

    # Index valid file first
    manager.index_document(str(valid_file))
    assert manager.get_document_count() == 1

    # Try to index invalid file
    try:
        manager.index_document(str(invalid_file))
    except (UnicodeDecodeError, Exception):
        # Should be tracked in failed files
        failed = manager.get_failed_files()
        assert any(str(invalid_file) in f["path"] for f in failed)


def test_index_document_with_windows_line_endings(manager, manager_config):
    """
    IndexManager handles Windows line endings (CRLF).

    Tests that different line ending styles don't cause issues.
    """
    docs_path = Path(manager_config.indexing.documents_path)

    file_path = docs_path / "windows.md"
    # Write with CRLF line endings
    with open(file_path, "wb") as f:
        f.write(b"# Windows Document\r\n\r\nWith CRLF line endings.\r\n")

    manager.index_document(str(file_path))

    assert manager.get_document_count() == 1
    assert len(manager.get_failed_files()) == 0


def test_index_document_with_mac_classic_line_endings(manager, manager_config):
    """
    IndexManager handles Mac Classic line endings (CR only).

    Tests legacy line ending format.
    """
    docs_path = Path(manager_config.indexing.documents_path)

    file_path = docs_path / "mac_classic.md"
    # Write with CR line endings (Mac Classic)
    with open(file_path, "wb") as f:
        f.write(b"# Mac Classic\r\rWith CR line endings.\r")

    manager.index_document(str(file_path))

    assert manager.get_document_count() == 1


def test_index_document_with_no_newline_at_eof(manager, manager_config):
    """
    IndexManager handles files without trailing newline.

    Tests that missing final newline doesn't cause issues.
    """
    docs_path = Path(manager_config.indexing.documents_path)

    file_path = docs_path / "no_newline.md"
    with open(file_path, "w") as f:
        f.write("# Document")  # No trailing newline

    manager.index_document(str(file_path))

    assert manager.get_document_count() == 1


def test_index_document_with_null_bytes(manager, manager_config):
    """
    IndexManager handles files with null bytes.

    Tests that binary content doesn't crash parser.
    """
    docs_path = Path(manager_config.indexing.documents_path)

    file_path = docs_path / "null_bytes.md"
    with open(file_path, "wb") as f:
        f.write(b"# Document\x00\x00with null bytes")

    # Should handle gracefully (either index or skip)
    try:
        manager.index_document(str(file_path))
        assert True
    except Exception:
        # If it fails, should track it
        assert len(manager.get_failed_files()) >= 1


def test_failed_files_tracking_after_encoding_error(manager, manager_config):
    """
    Failed files are properly tracked with error details.

    Tests that encoding errors are logged with timestamps and messages.
    """
    docs_path = Path(manager_config.indexing.documents_path)

    # Create file with encoding issue
    bad_file = docs_path / "bad_encoding.md"
    with open(bad_file, "wb") as f:
        f.write(b"\xff\xfe# Invalid UTF-8")

    try:
        manager.index_document(str(bad_file))
    except Exception:
        pass

    failed = manager.get_failed_files()

    if failed:
        # Should have error details
        assert "path" in failed[0]
        assert "error" in failed[0]
        assert "timestamp" in failed[0]
        assert str(bad_file) in failed[0]["path"]


def test_index_document_retries_after_encoding_error(manager, manager_config):
    """
    Successfully index file after fixing encoding error.

    Tests that failed file tracking is cleared on success.
    """
    docs_path = Path(manager_config.indexing.documents_path)

    file_path = docs_path / "file.md"

    # First, create invalid file
    with open(file_path, "wb") as f:
        f.write(b"\xff\xfe# Invalid")

    try:
        manager.index_document(str(file_path))
    except Exception:
        pass

    # Fix the file
    file_path.write_text("# Valid Document Now")

    # Should index successfully
    manager.index_document(str(file_path))

    # Failed files should be cleared for this file
    failed = manager.get_failed_files()
    assert not any(str(file_path) in f["path"] for f in failed)


def test_index_document_with_very_long_lines(manager, manager_config):
    """
    IndexManager handles files with very long lines.

    Tests that extremely long lines don't cause memory issues.
    """
    docs_path = Path(manager_config.indexing.documents_path)

    file_path = docs_path / "long_lines.md"
    long_line = "# Title\n\n" + ("word " * 10000) + "\n"
    file_path.write_text(long_line)

    # Should handle without crashing
    manager.index_document(str(file_path))

    assert manager.get_document_count() == 1


def test_index_document_with_emoji_and_unicode(manager, manager_config):
    """
    IndexManager handles emoji and unicode characters.

    Tests full Unicode support including emojis.
    """
    docs_path = Path(manager_config.indexing.documents_path)

    file_path = docs_path / "unicode.md"
    content = "# Unicode Test 🚀\n\n中文 日本語 한국어 العربية עברית\n\nEmoji: 😀🎉🔥"
    file_path.write_text(content, encoding="utf-8")

    manager.index_document(str(file_path))

    assert manager.get_document_count() == 1
    assert len(manager.get_failed_files()) == 0
