from pathlib import Path

import pytest
from searchkernel.api import (
    ActiveModelMetadata,
    CURRENT_MANIFEST_SPEC_VERSION,
    IndexManifest,
    MigrationPhase,
    MigrationState,
    ModelNamespace,
    ReindexError,
    save_manifest,
)

from mcp_markdown_ragdocs.config import Config
from mcp_markdown_ragdocs.indexing.reindex import (
    ManifestModelLifecycleStore,
    default_reindex_status,
    reindex_status_payload,
    run_reindex_operation,
    write_reindex_status,
)


def test_manifest_lifecycle_persists_active_model_and_checkpoint(tmp_path: Path):
    source = ModelNamespace("old-model", 2)
    target = ModelNamespace("new-model", 3)
    save_manifest(
        tmp_path,
        IndexManifest(
            spec_version=CURRENT_MANIFEST_SPEC_VERSION,
            embedding_model=source.model_name,
            chunking_config={},
            indexed_files={},
        ),
    )
    store = ManifestModelLifecycleStore(tmp_path, source_namespace=source)

    active = ActiveModelMetadata(target, generation=2, activated_at="now")
    assert store.compare_and_set_active_model(
        ActiveModelMetadata(source),
        active,
    )

    state = MigrationState(
        migration_id="reindex:new-model:3",
        source=source,
        target=target,
        phase=MigrationPhase.FLIP,
        checkpoint=4,
        total_records=4,
        corpus_fingerprint="fingerprint",
    )
    store.save_migration(state)

    loaded = reindex_status_payload(tmp_path, tmp_path)
    assert loaded["active_model"] == active.to_dict()
    assert loaded["migration"] == state.to_dict()
    assert loaded["phase"] == "flip"
    assert loaded["checkpoint"] == 4


def test_status_defaults_are_explicit_and_durable(tmp_path: Path):
    assert default_reindex_status()["status"] == "idle"
    write_reindex_status(tmp_path, {"status": "running", "phase": "backfill"})

    assert reindex_status_payload(tmp_path, tmp_path)["status"] == "running"


def test_legacy_chunk_backend_rejects_durable_migration(tmp_path: Path):
    with pytest.raises(ReindexError, match="legacy faiss\\+sqlite"):
        run_reindex_operation(
            config=Config(),
            index_path=tmp_path,
            runtime_root=tmp_path,
            operation="start",
            model="new-model",
            truncate_dim=None,
            old_model=None,
        )
