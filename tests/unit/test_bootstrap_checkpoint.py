from pathlib import Path

from src.indexing.bootstrap_checkpoint import (
    BootstrapCheckpoint,
    BootstrapFileStamp,
    compute_bootstrap_generation,
    load_bootstrap_checkpoint,
    mark_bootstrap_file_completed,
    prepare_bootstrap_checkpoint,
    save_bootstrap_checkpoint,
)
from src.indexing.manifest import CURRENT_MANIFEST_SPEC_VERSION, IndexManifest


def _manifest() -> IndexManifest:
    return IndexManifest(
        spec_version=CURRENT_MANIFEST_SPEC_VERSION,
        embedding_model="local",
        chunking_config={
            "strategy": "header",
            "min_chunk_chars": 200,
            "max_chunk_chars": 2000,
            "overlap_chars": 100,
        },
        indexed_files={},
    )


def test_prepare_bootstrap_checkpoint_resets_on_generation_mismatch(tmp_path: Path):
    manifest = _manifest()
    original_targets = {
        "docs/one.md": BootstrapFileStamp("docs/one.md", mtime_ns=1, size=10),
    }
    stale_checkpoint = BootstrapCheckpoint(
        schema_version="1.0.0",
        generation="stale-generation",
        complete=False,
        targets=original_targets,
        completed=original_targets,
    )
    save_bootstrap_checkpoint(tmp_path, stale_checkpoint)

    new_targets = {
        "docs/two.md": BootstrapFileStamp("docs/two.md", mtime_ns=2, size=20),
    }
    checkpoint = prepare_bootstrap_checkpoint(
        tmp_path,
        compute_bootstrap_generation(manifest, new_targets),
        new_targets,
    )

    assert checkpoint.targets == new_targets
    assert checkpoint.completed == {}
    assert checkpoint.complete is False


def test_mark_bootstrap_file_completed_updates_checkpoint(tmp_path: Path):
    docs_root = tmp_path / "docs"
    docs_root.mkdir()
    file_path = docs_root / "guide.md"
    file_path.write_text("# Guide")

    manifest = _manifest()
    target_stamp = BootstrapFileStamp(
        relative_path="guide.md",
        mtime_ns=0,
        size=0,
    )
    checkpoint = BootstrapCheckpoint(
        schema_version="1.0.0",
        generation=compute_bootstrap_generation(manifest, {"guide.md": target_stamp}),
        complete=False,
        targets={"guide.md": target_stamp},
        completed={},
    )
    save_bootstrap_checkpoint(tmp_path, checkpoint)

    marked = mark_bootstrap_file_completed(tmp_path, [docs_root], str(file_path))
    saved = load_bootstrap_checkpoint(tmp_path)

    assert marked is True
    assert saved is not None
    assert saved.complete is True
    assert set(saved.completed) == {"guide.md"}
    assert saved.completed["guide.md"].size == file_path.stat().st_size


def test_load_bootstrap_checkpoint_returns_none_for_corruption(tmp_path: Path):
    (tmp_path / "bootstrap.checkpoint.json").write_text("not-json")

    assert load_bootstrap_checkpoint(tmp_path) is None