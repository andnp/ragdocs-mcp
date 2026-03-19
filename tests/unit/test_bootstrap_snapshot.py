from __future__ import annotations

from pathlib import Path

from src.indexing.bootstrap_checkpoint import BootstrapCheckpoint, BootstrapFileStamp
from src.indexing.bootstrap_snapshot import (
    BootstrapReadinessSnapshot,
    PublicIndexStateSnapshot,
    compute_bootstrap_completed_paths,
    derive_bootstrap_readiness_snapshot,
    derive_loaded_index_state_snapshot,
)
from src.indexing.manifest import CURRENT_MANIFEST_SPEC_VERSION, IndexManifest


def _stamp(file_path: Path) -> BootstrapFileStamp:
    stat_result = file_path.stat()
    return BootstrapFileStamp(
        relative_path=file_path.name,
        mtime_ns=stat_result.st_mtime_ns,
        size=stat_result.st_size,
    )


def _manifest(indexed_files: dict[str, str]) -> IndexManifest:
    return IndexManifest(
        spec_version=CURRENT_MANIFEST_SPEC_VERSION,
        embedding_model="local",
        chunking_config={},
        indexed_files=indexed_files,
    )


def test_compute_bootstrap_completed_paths_intersects_checkpoint_manifest_and_targets(
    tmp_path: Path,
):
    doc_one = tmp_path / "doc1.md"
    doc_two = tmp_path / "doc2.md"
    doc_one.write_text("# Doc 1")
    doc_two.write_text("# Doc 2")

    doc_one_stamp = _stamp(doc_one)
    doc_two_stamp = _stamp(doc_two)
    target_stamps = {
        "doc1.md": doc_one_stamp,
        "doc2.md": doc_two_stamp,
    }
    checkpoint = BootstrapCheckpoint(
        schema_version="1.0.0",
        generation="generation",
        complete=False,
        targets=target_stamps,
        completed={
            "doc1.md": doc_one_stamp,
            "doc2.md": BootstrapFileStamp(
                relative_path="doc2.md",
                mtime_ns=doc_two_stamp.mtime_ns,
                size=doc_two_stamp.size + 1,
            ),
        },
    )

    completed_paths = compute_bootstrap_completed_paths(
        checkpoint,
        _manifest({"doc1": "doc1.md", "doc2": "doc2.md"}),
        target_stamps,
    )

    assert completed_paths == {"doc1.md"}


def test_derive_loaded_index_state_snapshot_marks_partial_until_all_targets_loaded():
    assert derive_loaded_index_state_snapshot(3, 2) == PublicIndexStateSnapshot(
        status="partial",
        indexed_count=2,
        total_count=3,
    )
    assert derive_loaded_index_state_snapshot(3, 3) == PublicIndexStateSnapshot(
        status="ready",
        indexed_count=3,
        total_count=3,
    )


def test_derive_bootstrap_readiness_snapshot_surfaces_partial_queryable_state(
    tmp_path: Path,
):
    doc_one = tmp_path / "doc1.md"
    doc_two = tmp_path / "doc2.md"
    doc_one.write_text("# Doc 1")
    doc_two.write_text("# Doc 2")

    target_stamps = {
        "doc1.md": _stamp(doc_one),
        "doc2.md": _stamp(doc_two),
    }
    checkpoint = BootstrapCheckpoint(
        schema_version="1.0.0",
        generation="generation",
        complete=False,
        targets=target_stamps,
        completed={
            "doc1.md": target_stamps["doc1.md"],
        },
    )

    snapshot = derive_bootstrap_readiness_snapshot(
        checkpoint,
        _manifest({"doc1": "doc1.md"}),
        target_stamps,
        loaded_indexed_count=1,
        queryable=True,
        rebuild_pending=False,
    )

    assert snapshot == BootstrapReadinessSnapshot(
        total_targets=2,
        durably_completed_targets=1,
        loaded_indexed_count=1,
        queryable=True,
        public_state=PublicIndexStateSnapshot(
            status="partial",
            indexed_count=1,
            total_count=2,
        ),
    )


def test_derive_bootstrap_readiness_snapshot_surfaces_complete_rebuild_as_indexing(
    tmp_path: Path,
):
    doc = tmp_path / "doc.md"
    doc.write_text("# Doc")

    snapshot = derive_bootstrap_readiness_snapshot(
        checkpoint=None,
        saved_manifest=_manifest({"doc": "doc.md"}),
        target_stamps={"doc.md": _stamp(doc)},
        loaded_indexed_count=1,
        queryable=True,
        rebuild_pending=True,
    )

    assert snapshot is not None
    assert snapshot.queryable is True
    assert snapshot.public_state == PublicIndexStateSnapshot(
        status="indexing",
        indexed_count=1,
        total_count=1,
    )


def test_derive_bootstrap_readiness_snapshot_requires_loaded_docs_for_nonempty_targets(
    tmp_path: Path,
):
    doc = tmp_path / "doc.md"
    doc.write_text("# Doc")

    snapshot = derive_bootstrap_readiness_snapshot(
        checkpoint=None,
        saved_manifest=_manifest({"doc": "doc.md"}),
        target_stamps={"doc.md": _stamp(doc)},
        loaded_indexed_count=0,
        queryable=False,
        rebuild_pending=False,
    )

    assert snapshot is None