"""
Unit tests for reconciliation logic.
"""

import pytest

from src.indexing.manifest import IndexManifest
from src.indexing.reconciler import (
    reconcile_indices,
    build_indexed_files_map,
    find_excluded_indexed_files,
)


@pytest.fixture
def docs_path(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    return docs


@pytest.fixture
def sample_manifest():
    # doc_id is now relative path without extension (e.g., "subdir/doc2" not "doc2")
    return IndexManifest(
        spec_version="1.0.0",
        embedding_model="local",
        parsers={"**/*.md": "MarkdownParser"},
        chunking_config={},
        indexed_files={
            "doc1": "doc1.md",
            "subdir/doc2": "subdir/doc2.md",
            "doc3": "doc3.md",
        }
    )


def test_reconcile_no_changes(docs_path, sample_manifest):
    """Test reconciliation when filesystem matches manifest."""
    # Create files matching manifest
    (docs_path / "doc1.md").write_text("# Doc 1")
    (docs_path / "subdir").mkdir()
    (docs_path / "subdir/doc2.md").write_text("# Doc 2")
    (docs_path / "doc3.md").write_text("# Doc 3")

    discovered = [
        str(docs_path / "doc1.md"),
        str(docs_path / "subdir/doc2.md"),
        str(docs_path / "doc3.md"),
    ]

    files_to_add, doc_ids_to_remove, _ = reconcile_indices(
        discovered,
        sample_manifest,
        docs_path
    )

    assert files_to_add == []
    assert doc_ids_to_remove == []


def test_reconcile_new_files(docs_path, sample_manifest):
    """Test reconciliation detects new files to index."""
    # Create original files
    (docs_path / "doc1.md").write_text("# Doc 1")
    (docs_path / "subdir").mkdir()
    (docs_path / "subdir/doc2.md").write_text("# Doc 2")
    (docs_path / "doc3.md").write_text("# Doc 3")

    # Add new files
    (docs_path / "new.md").write_text("# New")
    (docs_path / "subdir/new2.md").write_text("# New 2")

    discovered = [
        str(docs_path / "doc1.md"),
        str(docs_path / "subdir/doc2.md"),
        str(docs_path / "doc3.md"),
        str(docs_path / "new.md"),
        str(docs_path / "subdir/new2.md"),
    ]

    files_to_add, doc_ids_to_remove, _ = reconcile_indices(
        discovered,
        sample_manifest,
        docs_path
    )

    assert len(files_to_add) == 2
    assert str(docs_path / "new.md") in files_to_add
    assert str(docs_path / "subdir/new2.md") in files_to_add
    assert doc_ids_to_remove == []


def test_reconcile_deleted_files(docs_path, sample_manifest):
    """Test reconciliation detects deleted files to remove."""
    # Only create some of the files (doc2 and doc3 deleted)
    (docs_path / "doc1.md").write_text("# Doc 1")

    discovered = [
        str(docs_path / "doc1.md"),
    ]

    files_to_add, doc_ids_to_remove, _ = reconcile_indices(
        discovered,
        sample_manifest,
        docs_path
    )

    assert files_to_add == []
    assert len(doc_ids_to_remove) == 2
    assert "subdir/doc2" in doc_ids_to_remove
    assert "doc3" in doc_ids_to_remove


def test_reconcile_mixed_changes(docs_path, sample_manifest):
    """Test reconciliation with both additions and deletions."""
    # Keep doc1, delete doc2/doc3, add new files
    (docs_path / "doc1.md").write_text("# Doc 1")
    (docs_path / "new.md").write_text("# New")
    (docs_path / "another.md").write_text("# Another")

    discovered = [
        str(docs_path / "doc1.md"),
        str(docs_path / "new.md"),
        str(docs_path / "another.md"),
    ]

    files_to_add, doc_ids_to_remove, _ = reconcile_indices(
        discovered,
        sample_manifest,
        docs_path
    )

    assert len(files_to_add) == 2
    assert str(docs_path / "new.md") in files_to_add
    assert str(docs_path / "another.md") in files_to_add

    assert len(doc_ids_to_remove) == 2
    assert "subdir/doc2" in doc_ids_to_remove
    assert "doc3" in doc_ids_to_remove


def test_reconcile_empty_manifest(docs_path):
    """Test reconciliation with empty manifest (first run)."""
    manifest = IndexManifest(
        spec_version="1.0.0",
        embedding_model="local",
        parsers={},
        chunking_config={},
        indexed_files={}
    )

    (docs_path / "doc1.md").write_text("# Doc 1")
    (docs_path / "doc2.md").write_text("# Doc 2")

    discovered = [
        str(docs_path / "doc1.md"),
        str(docs_path / "doc2.md"),
    ]

    files_to_add, doc_ids_to_remove, _ = reconcile_indices(
        discovered,
        manifest,
        docs_path
    )

    assert len(files_to_add) == 2
    assert doc_ids_to_remove == []


def test_reconcile_manifest_none(docs_path):
    """Test reconciliation when manifest.indexed_files is None."""
    manifest = IndexManifest(
        spec_version="1.0.0",
        embedding_model="local",
        parsers={},
        chunking_config={},
        indexed_files=None
    )

    (docs_path / "doc1.md").write_text("# Doc 1")

    discovered = [str(docs_path / "doc1.md")]

    files_to_add, doc_ids_to_remove, _ = reconcile_indices(
        discovered,
        manifest,
        docs_path
    )

    assert len(files_to_add) == 1
    assert doc_ids_to_remove == []


def test_build_indexed_files_map(docs_path):
    """Test building indexed_files map from file list."""
    (docs_path / "doc1.md").write_text("# Doc 1")
    (docs_path / "subdir").mkdir()
    (docs_path / "subdir/doc2.md").write_text("# Doc 2")

    files = [
        str(docs_path / "doc1.md"),
        str(docs_path / "subdir/doc2.md"),
    ]

    indexed_map = build_indexed_files_map(files, docs_path)

    # doc_id is now relative path without extension
    assert indexed_map == {
        "doc1": "doc1.md",
        "subdir/doc2": "subdir/doc2.md",
    }


def test_build_indexed_files_map_empty(docs_path):
    """Test building indexed_files map with no files."""
    indexed_map = build_indexed_files_map([], docs_path)
    assert indexed_map == {}


def test_reconcile_ignores_files_outside_docs_path(docs_path, tmp_path):
    """Test that files outside docs_path are ignored."""
    outside_file = tmp_path / "outside.md"
    outside_file.write_text("# Outside")

    (docs_path / "doc1.md").write_text("# Doc 1")

    manifest = IndexManifest(
        spec_version="1.0.0",
        embedding_model="local",
        parsers={},
        chunking_config={},
        indexed_files={"doc1": "doc1.md"}
    )

    # Include outside file in discovered (should be ignored)
    discovered = [
        str(docs_path / "doc1.md"),
        str(outside_file),
    ]

    files_to_add, doc_ids_to_remove, _ = reconcile_indices(
        discovered,
        manifest,
        docs_path
    )

    # Outside file should not be added
    assert files_to_add == []
    assert doc_ids_to_remove == []


def test_find_excluded_indexed_files_venv(docs_path):
    """Test that files in .venv are detected as excluded."""
    manifest = IndexManifest(
        spec_version="1.0.0",
        embedding_model="local",
        parsers={},
        chunking_config={},
        indexed_files={
            "normal": "normal.md",
            ".venv/lib/package/README": ".venv/lib/package/README.md",
            "docs/guide": "docs/guide.md",
        }
    )

    excluded = find_excluded_indexed_files(
        manifest,
        docs_path,
        include_patterns=["**/*"],
        exclude_patterns=["**/.venv/**"],
        exclude_hidden_dirs=True,
    )

    # .venv file should be detected as excluded
    assert ".venv/lib/package/README" in excluded
    assert "normal" not in excluded
    assert "docs/guide" not in excluded


def test_find_excluded_indexed_files_hidden_dirs(docs_path):
    """Test that files in hidden directories are detected as excluded."""
    manifest = IndexManifest(
        spec_version="1.0.0",
        embedding_model="local",
        parsers={},
        chunking_config={},
        indexed_files={
            "normal": "normal.md",
            ".hidden/secret": ".hidden/secret.md",
            ".git/config": ".git/config.md",
        }
    )

    excluded = find_excluded_indexed_files(
        manifest,
        docs_path,
        include_patterns=["**/*"],
        exclude_patterns=[],
        exclude_hidden_dirs=True,
    )

    # Hidden directory files should be excluded
    assert ".hidden/secret" in excluded
    assert ".git/config" in excluded
    assert "normal" not in excluded


def test_find_excluded_indexed_files_multiple_patterns(docs_path):
    """Test with multiple exclude patterns."""
    manifest = IndexManifest(
        spec_version="1.0.0",
        embedding_model="local",
        parsers={},
        chunking_config={},
        indexed_files={
            "normal": "normal.md",
            "build/output": "build/output.md",
            "node_modules/package/README": "node_modules/package/README.md",
            ".venv/lib/README": ".venv/lib/README.md",
        }
    )

    excluded = find_excluded_indexed_files(
        manifest,
        docs_path,
        include_patterns=["**/*"],
        exclude_patterns=["**/build/**", "**/node_modules/**", "**/.venv/**"],
        exclude_hidden_dirs=True,
    )

    assert "build/output" in excluded
    assert "node_modules/package/README" in excluded
    assert ".venv/lib/README" in excluded
    assert "normal" not in excluded


def test_reconcile_with_exclude_patterns(docs_path):
    """Test full reconciliation with exclude patterns."""
    # Create a manifest with some files that would now be excluded
    manifest = IndexManifest(
        spec_version="1.0.0",
        embedding_model="local",
        parsers={},
        chunking_config={},
        indexed_files={
            "doc1": "doc1.md",
            ".venv/lib/README": ".venv/lib/README.md",
            "build/docs/api": "build/docs/api.md",
        }
    )

    # Create only the valid file
    (docs_path / "doc1.md").write_text("# Doc 1")

    # discovered_files would not include .venv or build due to filtering
    discovered = [str(docs_path / "doc1.md")]

    files_to_add, doc_ids_to_remove, _ = reconcile_indices(
        discovered,
        manifest,
        docs_path,
        include_patterns=["**/*"],
        exclude_patterns=["**/.venv/**", "**/build/**"],
        exclude_hidden_dirs=True,
    )

    # Both excluded files should be in removal list
    assert ".venv/lib/README" in doc_ids_to_remove
    assert "build/docs/api" in doc_ids_to_remove
    assert "doc1" not in doc_ids_to_remove
    assert files_to_add == []


def test_reconcile_blacklist_change_removes_indexed_files(docs_path):
    """Test that changing blacklist config removes previously indexed files.

    Simulates scenario: user adds '**/.venv/**' to exclude patterns after
    accidentally indexing .venv files.
    """
    manifest = IndexManifest(
        spec_version="1.0.0",
        embedding_model="local",
        parsers={},
        chunking_config={},
        indexed_files={
            "docs/README": "docs/README.md",
            ".venv/lib/python/site-packages/flax/README": ".venv/lib/python/site-packages/flax/README.md",
            ".venv/lib/python/site-packages/orbax/README": ".venv/lib/python/site-packages/orbax/README.md",
        }
    )

    # Only docs file exists in discovered (venv filtered out by discovery)
    (docs_path / "docs").mkdir()
    (docs_path / "docs/README.md").write_text("# Docs")

    discovered = [str(docs_path / "docs/README.md")]

    files_to_add, doc_ids_to_remove, _ = reconcile_indices(
        discovered,
        manifest,
        docs_path,
        include_patterns=["**/*"],
        exclude_patterns=["**/.venv/**"],
        exclude_hidden_dirs=True,
    )

    # Both .venv files should be removed
    assert len(doc_ids_to_remove) == 2
    assert ".venv/lib/python/site-packages/flax/README" in doc_ids_to_remove
    assert ".venv/lib/python/site-packages/orbax/README" in doc_ids_to_remove
    assert files_to_add == []
