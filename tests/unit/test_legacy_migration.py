"""Tests for src/indexing/migration.py – legacy index artefact detection and cleanup."""

from __future__ import annotations

from pathlib import Path


from src.indexing.migration import detect_and_migrate_legacy_index, _is_whoosh_directory
from src.indexing.manifest import IndexManifest, save_manifest, load_manifest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_whoosh_dir(path: Path) -> None:
    """Create a directory that looks like an old Whoosh keyword index."""
    path.mkdir(parents=True, exist_ok=True)
    # Whoosh writes files named MAIN_*.seg and _MAIN_*.toc
    (path / "MAIN_vv0e9gps33tddiv2.seg").write_bytes(b"dummy whoosh segment")
    (path / "_MAIN_1.toc").write_bytes(b"dummy whoosh toc")


def _make_sqlite_keyword_dir(path: Path) -> None:
    """Create a directory that already contains a modern SQLite keyword index."""
    path.mkdir(parents=True, exist_ok=True)
    (path / "index.db").write_bytes(b"SQLite format 3")


def _make_snapshots_dir(path: Path) -> None:
    """Create a legacy snapshots directory tree."""
    (path / "v1").mkdir(parents=True, exist_ok=True)
    (path / "v1" / "complete.marker").write_text("done")
    (path / "v2").mkdir(parents=True, exist_ok=True)
    (path / "version.bin").write_bytes(b"\x00\x00\x00\x02")


def _save_manifest_with_files(index_path: Path, files: dict[str, str]) -> None:
    manifest = IndexManifest(
        spec_version="1.0.0",
        embedding_model="test-model",
        chunking_config={},
        indexed_files=files,
    )
    save_manifest(index_path, manifest)


# ---------------------------------------------------------------------------
# _is_whoosh_directory
# ---------------------------------------------------------------------------


class TestIsWhooshDirectory:
    def test_returns_false_for_nonexistent(self, tmp_path: Path) -> None:
        assert not _is_whoosh_directory(tmp_path / "no_such_dir")

    def test_returns_false_for_file(self, tmp_path: Path) -> None:
        f = tmp_path / "file.txt"
        f.write_text("hi")
        assert not _is_whoosh_directory(f)

    def test_returns_false_for_sqlite_dir(self, tmp_path: Path) -> None:
        p = tmp_path / "keyword"
        _make_sqlite_keyword_dir(p)
        assert not _is_whoosh_directory(p)

    def test_returns_true_for_whoosh_dir(self, tmp_path: Path) -> None:
        p = tmp_path / "keyword"
        _make_whoosh_dir(p)
        assert _is_whoosh_directory(p)

    def test_returns_false_for_empty_dir(self, tmp_path: Path) -> None:
        p = tmp_path / "keyword"
        p.mkdir()
        assert not _is_whoosh_directory(p)


# ---------------------------------------------------------------------------
# detect_and_migrate_legacy_index
# ---------------------------------------------------------------------------


class TestDetectAndMigrateLegacyIndex:
    def test_no_op_when_no_legacy_artefacts(self, tmp_path: Path) -> None:
        result = detect_and_migrate_legacy_index(tmp_path)
        assert result is False

    def test_removes_snapshots_directory(self, tmp_path: Path) -> None:
        snapshots = tmp_path / "snapshots"
        _make_snapshots_dir(snapshots)
        assert snapshots.is_dir()

        result = detect_and_migrate_legacy_index(tmp_path)

        assert result is True
        assert not snapshots.exists()

    def test_removes_whoosh_keyword_directory(self, tmp_path: Path) -> None:
        keyword = tmp_path / "keyword"
        _make_whoosh_dir(keyword)

        result = detect_and_migrate_legacy_index(tmp_path)

        assert result is True
        assert not keyword.exists()

    def test_resets_manifest_indexed_files_when_whoosh_removed(self, tmp_path: Path) -> None:
        keyword = tmp_path / "keyword"
        _make_whoosh_dir(keyword)
        _save_manifest_with_files(tmp_path, {"doc1": "README.md", "doc2": "guide.md"})

        detect_and_migrate_legacy_index(tmp_path)

        manifest = load_manifest(tmp_path)
        assert manifest is not None
        assert manifest.indexed_files == {}

    def test_does_not_reset_manifest_for_snapshots_only(self, tmp_path: Path) -> None:
        snapshots = tmp_path / "snapshots"
        _make_snapshots_dir(snapshots)
        _save_manifest_with_files(tmp_path, {"doc1": "README.md"})

        detect_and_migrate_legacy_index(tmp_path)

        manifest = load_manifest(tmp_path)
        assert manifest is not None
        # indexed_files must not have been cleared – no keyword migration happened
        assert manifest.indexed_files == {"doc1": "README.md"}

    def test_does_not_touch_sqlite_keyword_directory(self, tmp_path: Path) -> None:
        keyword = tmp_path / "keyword"
        _make_sqlite_keyword_dir(keyword)
        _save_manifest_with_files(tmp_path, {"doc1": "README.md"})

        result = detect_and_migrate_legacy_index(tmp_path)

        assert result is False
        assert keyword.is_dir()
        assert (keyword / "index.db").exists()

    def test_handles_both_artefacts_together(self, tmp_path: Path) -> None:
        _make_whoosh_dir(tmp_path / "keyword")
        _make_snapshots_dir(tmp_path / "snapshots")
        _save_manifest_with_files(tmp_path, {"doc1": "README.md"})

        result = detect_and_migrate_legacy_index(tmp_path)

        assert result is True
        assert not (tmp_path / "keyword").exists()
        assert not (tmp_path / "snapshots").exists()
        manifest = load_manifest(tmp_path)
        assert manifest is not None
        assert manifest.indexed_files == {}

    def test_ok_when_no_manifest(self, tmp_path: Path) -> None:
        """Migration must not crash when there is no saved manifest."""
        _make_whoosh_dir(tmp_path / "keyword")

        # Should not raise
        result = detect_and_migrate_legacy_index(tmp_path)
        assert result is True
