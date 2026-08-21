"""Tests for the persisted startup manifest boundary."""

from pathlib import Path

from searchkernel.api import (
    ActiveModelMetadata,
    CURRENT_MANIFEST_SPEC_VERSION,
    IndexManifest,
    MigrationPhase,
    MigrationState,
    ModelNamespace,
    load_manifest,
    save_manifest,
)

from mcp_markdown_ragdocs.app.bootstrap_manifest import ManifestCoordinator
from mcp_markdown_ragdocs.config import Config


class ManifestTestHost:
    """Small host implementing the manifest coordinator contract."""

    config: Config
    index_manager: object
    index_path: Path
    current_manifest: IndexManifest | None
    _is_virgin_startup: bool

    def __init__(self, index_path: Path) -> None:
        self.config = Config()
        self.index_manager = object()
        self.index_path = index_path
        self.current_manifest = None
        self._is_virgin_startup = False


def test_fresh_task_bootstrap_persists_manifest_for_restart(tmp_path: Path) -> None:
    """
    A fresh bootstrap writes its manifest before task indexing begins.

    After the task records indexed-file membership, a new startup observes
    the same manifest and does not request a full rebuild.
    """
    host = ManifestTestHost(tmp_path)

    assert ManifestCoordinator(host).check_and_rebuild_if_needed() is True
    initial = load_manifest(tmp_path)

    assert initial is not None
    assert initial.indexed_files == {}

    initial.indexed_files = {"guide": "guide.md"}
    save_manifest(tmp_path, initial)

    restarted = ManifestTestHost(tmp_path)
    assert ManifestCoordinator(restarted).check_and_rebuild_if_needed() is False
    restarted_manifest = load_manifest(tmp_path)
    assert restarted_manifest is not None
    assert restarted_manifest.indexed_files == {"guide": "guide.md"}


def test_startup_manifest_preserves_model_migration_metadata(tmp_path: Path) -> None:
    """
    Creating the next startup manifest retains active model migration state.

    Task-backed startup must not erase metadata needed by model lifecycle
    operations while it creates or refreshes indexed-file membership.
    """
    source = ModelNamespace("old-model", 2)
    target = ModelNamespace("new-model", 3)
    saved = IndexManifest(
        spec_version=CURRENT_MANIFEST_SPEC_VERSION,
        embedding_model="old-model",
        chunking_config={},
        indexed_files={"guide": "guide.md"},
        active_model=ActiveModelMetadata(source, generation=4),
        migration=MigrationState(
            "migration-1", source, target, phase=MigrationPhase.BACKFILL
        ),
    )
    save_manifest(tmp_path, saved)
    host = ManifestTestHost(tmp_path)

    ManifestCoordinator(host).check_and_rebuild_if_needed()
    current = host.current_manifest

    assert current is not None
    assert current.active_model == saved.active_model
    assert current.migration == saved.migration
