from pathlib import Path
from typing import cast

import pytest
from searchkernel.api import (
    ActiveModelMetadata,
    CURRENT_MANIFEST_SPEC_VERSION,
    EmbeddingProvider,
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
    _DeferredEmbeddingProvider,
    PgvectorReindexStore,
    build_embedding_provider,
    default_reindex_status,
    read_reindex_status,
    reindex_status_payload,
    run_reindex_operation,
    submit_reindex_status,
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


def test_manifest_lifecycle_rejects_missing_namespace_store_and_stale_cas(
    tmp_path: Path,
) -> None:
    source = ModelNamespace("old-model", 2)
    target = ModelNamespace("new-model", 3)
    store = ManifestModelLifecycleStore(tmp_path, source_namespace=source)

    with pytest.raises(ReindexError, match="namespace storage"):
        store.ensure_namespace(target)
    assert store.compare_and_set_active_model(ActiveModelMetadata(target), ActiveModelMetadata(source)) is False
    assert store.load_migration("missing") is None


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


def test_reindex_status_recovers_from_invalid_payload(tmp_path: Path):
    status_path = tmp_path / "reindex-status.json"
    status_path.write_text("[]", encoding="utf-8")
    assert read_reindex_status(tmp_path)["status"] == "idle"

    status_path.write_text("{invalid", encoding="utf-8")
    assert read_reindex_status(tmp_path)["phase"] == "idle"

    queued = submit_reindex_status(
        tmp_path,
        operation="start",
        request_id="request-1",
        model="new-model",
        truncate_dim=1024,
        old_model="old-model",
    )
    assert queued["status"] == "queued"
    assert queued["request_id"] == "request-1"


def test_reindex_provider_and_namespace_helpers_validate_inputs(tmp_path: Path):
    config = Config()
    config.embedding.provider = "huggingface"
    with pytest.raises(ReindexError, match="unsupported reindex embedding provider"):
        build_embedding_provider(config, "new-model", None)

    with pytest.raises(ReindexError, match="requires store.pg_dsn"):
        PgvectorReindexStore("")

    assert PgvectorReindexStore._sanitize_model_name("BAAI/bge-small") != ""
    assert PgvectorReindexStore._vector_literal([1, 2.5]) == "[1.0,2.5]"

    class _Provider:
        model_name = "new-model"
        dim = 2

        def embed(self, texts: list[str]) -> list[list[float]]:
            return [[float(len(texts))] * 2]

    provider = _DeferredEmbeddingProvider(
        "new-model",
        2,
        lambda: cast(EmbeddingProvider, _Provider()),
    )
    assert provider.embed(["one"]) == [[1.0, 1.0]]
