"""
Unit tests for reconciliation logic.
"""

import pytest

from src.indexing.manifest import IndexManifest
from src.indexing.reconciler import reconcile_indices, build_indexed_files_map


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

    files_to_add, doc_ids_to_remove = reconcile_indices(
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

    files_to_add, doc_ids_to_remove = reconcile_indices(
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

    files_to_add, doc_ids_to_remove = reconcile_indices(
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

    files_to_add, doc_ids_to_remove = reconcile_indices(
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

    files_to_add, doc_ids_to_remove = reconcile_indices(
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

    files_to_add, doc_ids_to_remove = reconcile_indices(
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

    files_to_add, doc_ids_to_remove = reconcile_indices(
        discovered,
        manifest,
        docs_path
    )

    # Outside file should not be added
    assert files_to_add == []
    assert doc_ids_to_remove == []
