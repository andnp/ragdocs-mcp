from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from click.testing import CliRunner
from huey import SqliteHuey
from searchkernel.api import (
    ActiveModelMetadata,
    BackupMetadata,
    CURRENT_MANIFEST_SPEC_VERSION,
    IndexManifest,
    MigrationPhase,
    ModelDimensionMismatchError,
    ModelNamespace,
    ReindexError,
    ReindexRoutine,
    RecordBatch,
    Record,
    RecordStatus,
    RollbackMetadata,
    TaskSubmissionResult,
    ValidationResult,
    load_manifest,
    save_manifest,
)
from searchkernel.domain import RecordHit, RecordIdentity, SearchFilters, Vector

import mcp_markdown_ragdocs.indexing.reindex as reindex_module
import mcp_markdown_ragdocs.indexing.tasks as tasks_module
from mcp_markdown_ragdocs.cli import cli
from mcp_markdown_ragdocs.coordination.task_leases import TaskLeaseStore
from mcp_markdown_ragdocs.coordination.work_intents import WorkIntentStore
from mcp_markdown_ragdocs.config import (
    Config,
    EmbeddingConfig,
    IndexingConfig,
    StoreConfig,
)
from mcp_markdown_ragdocs.daemon.request_router import (
    DaemonRequestRouterDependencies,
    build_daemon_request_handler,
)
from mcp_markdown_ragdocs.indexing.reindex import (
    ManifestModelLifecycleStore,
    reindex_status_payload,
    run_reindex_operation,
)
from mcp_markdown_ragdocs.indexing.tasks import register_tasks

pytestmark = pytest.mark.integration


@dataclass
class _RecordVectorStore:
    namespaces: dict[ModelNamespace, dict[str, Record]] = field(default_factory=dict)
    backups: dict[str, dict[str, Record]] = field(default_factory=dict)
    backup_namespaces: dict[str, ModelNamespace] = field(default_factory=dict)
    validation_errors: tuple[str, ...] = ()
    backup_seen_before_delete: bool = False

    def resolve_namespace(self, model_name: str) -> ModelNamespace:
        namespaces = [namespace for namespace in self.namespaces if namespace.model_name == model_name]
        if not namespaces:
            raise ReindexError(f"no namespace for {model_name!r}")
        dimensions = {namespace.dim for namespace in namespaces}
        if len(dimensions) != 1:
            raise ModelDimensionMismatchError(
                f"model {model_name!r} has mixed dimensions"
            )
        return namespaces[0]

    def ensure_namespace(self, namespace: ModelNamespace) -> None:
        existing = [
            current for current in self.namespaces if current.model_name == namespace.model_name
        ]
        if any(current.dim != namespace.dim for current in existing):
            raise ModelDimensionMismatchError(
                f"model {namespace.model_name!r} already has another dimension"
            )
        self.namespaces.setdefault(namespace, {})

    def delete_namespace(self, namespace: ModelNamespace) -> None:
        self.backup_seen_before_delete = namespace in self.backup_namespaces.values()
        self.namespaces.pop(namespace, None)

    def upsert(self, records: list[Record], model_name: str, dim: int) -> None:
        namespace = ModelNamespace(model_name, dim)
        self.ensure_namespace(namespace)
        for record in records:
            if record.embedding is None or len(record.embedding) != dim:
                raise ReindexError(f"record {record.storage_key} has wrong dimension")
            self.namespaces[namespace][record.storage_key] = deepcopy(record)

    def load_records(self, namespace: ModelNamespace) -> list[Record]:
        return list(self.namespaces.get(namespace, {}).values())

    def record_source(self, namespace: ModelNamespace) -> _RecordSource:
        return _RecordSource(self, namespace)

    def search(
        self,
        query_vector: Vector,
        k: int,
        *,
        model_name: str,
        dim: int,
        filters: SearchFilters | None = None,
    ) -> list[RecordHit]:
        del query_vector, filters
        namespace = ModelNamespace(model_name, dim)
        return [
            RecordHit(RecordIdentity.from_storage_key(storage_key), 1.0)
            for storage_key in list(self.namespaces.get(namespace, {}))[:k]
        ]

    def delete(self, record_ids: list[str]) -> None:
        for records in self.namespaces.values():
            for record_id in record_ids:
                records.pop(record_id, None)

    def epoch(self) -> int:
        return 0

    def validate_namespace(
        self,
        namespace: ModelNamespace,
        expected_records: int,
    ) -> ValidationResult:
        return ValidationResult(
            namespace=namespace,
            expected_records=expected_records,
            indexed_records=len(self.namespaces.get(namespace, {})),
            errors=self.validation_errors,
            checked_at="test",
        )

    def create_backup(self, namespace: ModelNamespace) -> BackupMetadata:
        backup_id = f"backup-{len(self.backups) + 1}"
        self.backups[backup_id] = deepcopy(self.namespaces.get(namespace, {}))
        self.backup_namespaces[backup_id] = namespace
        return BackupMetadata(
            backup_id=backup_id,
            namespace=namespace,
            reference=backup_id,
            created_at="test",
        )

    def restore_backup(self, backup: BackupMetadata) -> RollbackMetadata:
        self.namespaces[backup.namespace] = deepcopy(self.backups[backup.reference])
        return RollbackMetadata(
            from_namespace=backup.namespace,
            to_namespace=backup.namespace,
            backup_id=backup.backup_id,
            reason="test restore",
            rolled_back_at="test",
        )


class _RecordSource:
    def __init__(self, store: _RecordVectorStore, namespace: ModelNamespace) -> None:
        self._records = store.load_records(namespace)

    @property
    def total_records(self) -> int:
        return len(self._records)

    def fetch_batch(self, cursor: str | None, limit: int) -> RecordBatch:
        if limit < 1:
            raise ValueError("limit must be >= 1")
        remaining = [
            record
            for record in self._records
            if cursor is None or record.storage_key > cursor
        ]
        records = remaining[:limit]
        next_cursor = (
            records[-1].storage_key
            if len(records) == limit and len(remaining) > limit
            else None
        )
        return RecordBatch(
            records=records,
            next_cursor=next_cursor,
            total_records=self.total_records,
        )


class _Provider:
    def __init__(
        self,
        model_name: str,
        dim: int,
        *,
        fail_on_call: int | None = None,
    ) -> None:
        self.model_name = model_name
        self.dim = dim
        self.fail_on_call = fail_on_call
        self.calls = 0

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        if self.fail_on_call == self.calls:
            raise RuntimeError("embedding provider failed")
        return [[float(index + 1)] * self.dim for index, _text in enumerate(texts)]

    def embed_query(self, text: str) -> list[float]:
        del text
        return [1.0] * self.dim


def _record(source_id: str, dim: int) -> Record:
    now = datetime(2026, 8, 1, tzinfo=UTC)
    return Record(
        workspace_id="acceptance",
        source_kind="chunk",
        source_id=source_id,
        title=source_id,
        body=f"Body for {source_id}",
        created_at=now,
        updated_at=now,
        status=RecordStatus.ACTIVE,
        embedding=[1.0] * dim,
        embedding_model="old-model",
    )


def _config(tmp_path: Path, *, backend: str = "pgvector") -> Config:
    return Config(
        indexing=IndexingConfig(
            documents_path=str(tmp_path / "docs"),
            index_path=str(tmp_path / "index"),
        ),
        store=StoreConfig(backend=backend, pg_dsn="test-dsn"),
        embedding=EmbeddingConfig(batch_size=1),
    )


def _seed(
    tmp_path: Path,
    store: _RecordVectorStore,
    *,
    source: ModelNamespace = ModelNamespace("old-model", 768),
    records: list[Record] | None = None,
) -> list[Record]:
    index_path = tmp_path / "index"
    index_path.mkdir(parents=True, exist_ok=True)
    store.ensure_namespace(source)
    seeded = records or [_record("chunk-1", source.dim), _record("chunk-2", source.dim)]
    store.upsert(seeded, source.model_name, source.dim)
    save_manifest(
        index_path,
        IndexManifest(
            spec_version=CURRENT_MANIFEST_SPEC_VERSION,
            embedding_model=source.model_name,
            chunking_config={},
            indexed_files={},
            active_model=ActiveModelMetadata(source),
        ),
    )
    return seeded


def _routine(
    tmp_path: Path,
    store: _RecordVectorStore,
    provider: _Provider,
    records: list[Record],
    *,
    source: ModelNamespace = ModelNamespace("old-model", 768),
    migration_id: str = "migration:test",
) -> ReindexRoutine:
    lifecycle = ManifestModelLifecycleStore(
        tmp_path / "index",
        source_namespace=source,
        namespace_store=store,
    )
    return ReindexRoutine(
        records,
        provider,
        store,
        batch_size=1,
        lifecycle_store=lifecycle,
        migration_id=migration_id,
        source_namespace=source,
    )


def _patch_app_store(
    monkeypatch: pytest.MonkeyPatch,
    store: _RecordVectorStore,
    provider: _Provider,
) -> None:
    monkeypatch.setattr(reindex_module, "PgvectorReindexStore", lambda _dsn: store)
    monkeypatch.setattr(
        reindex_module,
        "build_embedding_provider",
        lambda _config, _model, _truncate_dim: provider,
    )


def _active_namespace(index_path: Path) -> ModelNamespace:
    manifest = load_manifest(index_path)
    assert manifest is not None
    assert manifest.active_model is not None
    return manifest.active_model.namespace


def test_record_backed_768_to_1024_keeps_old_model_queryable_during_backfill(
    tmp_path: Path,
) -> None:
    store = _RecordVectorStore()
    records = _seed(tmp_path, store)
    routine = _routine(tmp_path, store, _Provider("new-model", 1024), records)

    routine.expand()
    assert store.search([1.0] * 768, 1, model_name="old-model", dim=768)
    routine.backfill()

    assert len(store.load_records(ModelNamespace("new-model", 1024))) == len(records)
    assert store.search([1.0] * 768, 1, model_name="old-model", dim=768)
    assert routine.state is not None
    assert routine.state.phase is MigrationPhase.BACKFILL
    assert store.namespaces[ModelNamespace("old-model", 768)]


def test_reindex_checkpoint_restart_resumes_remaining_records(tmp_path: Path) -> None:
    store = _RecordVectorStore()
    records = _seed(tmp_path, store)
    provider = _Provider("new-model", 1024, fail_on_call=2)
    routine = _routine(tmp_path, store, provider, records)
    routine.expand()

    with pytest.raises(ReindexError, match="embedding provider failed"):
        routine.backfill()

    failed = routine.state
    assert failed is not None
    assert failed.phase is MigrationPhase.FAILED
    assert failed.resume_phase is MigrationPhase.BACKFILL
    assert failed.checkpoint == 1
    assert len(store.load_records(ModelNamespace("new-model", 1024))) == 1

    provider.fail_on_call = None
    restarted = _routine(
        tmp_path,
        store,
        provider,
        records,
        migration_id="migration:test",
    )
    assert restarted.retry().phase is MigrationPhase.BACKFILL
    progress = restarted.backfill()

    assert progress.checkpoint == len(records)
    assert restarted.state is not None
    assert restarted.state.phase is MigrationPhase.BACKFILL
    assert len(store.load_records(ModelNamespace("new-model", 1024))) == len(records)


def test_mixed_dimension_model_is_rejected(tmp_path: Path) -> None:
    store = _RecordVectorStore()
    records = _seed(tmp_path, store)
    routine = _routine(
        tmp_path,
        store,
        _Provider("old-model", 1024),
        records,
        migration_id="migration:mixed-dimension",
    )

    with pytest.raises(ModelDimensionMismatchError, match="cannot migrate"):
        routine.expand()

    assert ModelNamespace("old-model", 1024) not in store.namespaces


def test_flip_is_gated_by_successful_validation(tmp_path: Path) -> None:
    store = _RecordVectorStore(validation_errors=("missing target record",))
    records = _seed(tmp_path, store)
    routine = _routine(tmp_path, store, _Provider("new-model", 1024), records)
    routine.expand()
    routine.backfill()

    with pytest.raises(ReindexError, match="validation failed"):
        routine.validate()
    assert _active_namespace(tmp_path / "index") == ModelNamespace("old-model", 768)

    store.validation_errors = ()
    assert routine.retry().phase is MigrationPhase.VALIDATE
    routine.validate()
    flipped = routine.flip()
    assert flipped.phase is MigrationPhase.FLIP
    assert _active_namespace(tmp_path / "index") == ModelNamespace("new-model", 1024)


def test_rollback_before_flip_removes_only_target_namespace(tmp_path: Path) -> None:
    store = _RecordVectorStore()
    records = _seed(tmp_path, store)
    routine = _routine(tmp_path, store, _Provider("new-model", 1024), records)
    routine.expand()
    routine.backfill()

    rolled_back = routine.rollback("old-model")

    assert rolled_back.phase is MigrationPhase.ROLLBACK
    assert _active_namespace(tmp_path / "index") == ModelNamespace("old-model", 768)
    assert store.load_records(ModelNamespace("old-model", 768))
    assert not store.load_records(ModelNamespace("new-model", 1024))


def test_contract_creates_backup_before_cleanup_and_rollback_restores_source(
    tmp_path: Path,
) -> None:
    store = _RecordVectorStore()
    records = _seed(tmp_path, store)
    routine = _routine(tmp_path, store, _Provider("new-model", 1024), records)
    routine.run()
    complete = routine.contract("old-model")

    assert complete.phase is MigrationPhase.COMPLETE
    assert complete.backup is not None
    assert store.backup_seen_before_delete
    assert not store.load_records(ModelNamespace("old-model", 768))
    assert _active_namespace(tmp_path / "index") == ModelNamespace("new-model", 1024)

    rolled_back = routine.rollback("old-model")
    assert rolled_back.rollback is not None
    assert store.load_records(ModelNamespace("old-model", 768))
    assert not store.load_records(ModelNamespace("new-model", 1024))
    assert _active_namespace(tmp_path / "index") == ModelNamespace("old-model", 768)


def test_run_reindex_reports_active_model_manifest_and_status(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _RecordVectorStore()
    _seed(tmp_path, store)
    provider = _Provider("new-model", 1024)
    _patch_app_store(monkeypatch, store, provider)

    state = run_reindex_operation(
        config=_config(tmp_path),
        index_path=tmp_path / "index",
        runtime_root=tmp_path / "runtime",
        operation="start",
        model="new-model",
        truncate_dim=None,
        old_model=None,
    )
    payload = reindex_status_payload(tmp_path / "runtime", tmp_path / "index")
    active_model = cast(dict[str, object], payload["active_model"])
    migration = cast(dict[str, object], payload["migration"])

    assert state.phase is MigrationPhase.FLIP
    assert payload["status"] == "succeeded"
    assert active_model == {
        "namespace": {"model_name": "new-model", "dim": 1024},
        "generation": 1,
        "activated_at": active_model["activated_at"],
    }
    assert migration["phase"] == "flip"
    assert payload["phase"] == "flip"
    assert payload["checkpoint"] == len(store.load_records(ModelNamespace("new-model", 1024)))


def _router_dependencies(ctx: object, runtime_root: Path) -> DaemonRequestRouterDependencies:
    return DaemonRequestRouterDependencies(
        ctx=ctx,
        coordinator=SimpleNamespace(),
        runtime_root=runtime_root,
        queue_db_path=runtime_root / "queue.db",
        socket_path=runtime_root / "daemon.sock",
        index_db_path=runtime_root / "index.db",
        get_worker_running=lambda: True,
        get_worker_pid=lambda: 123,
        build_admin_overview_payload=lambda *_args, **_kwargs: {},
        build_index_stats_payload=lambda *_args, **_kwargs: {},
        build_queue_status_payload=lambda *_args, **_kwargs: {},
    )


def _router_context(tmp_path: Path, backend: str) -> SimpleNamespace:
    return SimpleNamespace(
        config=SimpleNamespace(
            store=SimpleNamespace(backend=backend),
            indexing=SimpleNamespace(task_backpressure_limit=5),
        ),
        index_path=tmp_path / "index",
    )


@pytest.mark.asyncio
async def test_daemon_rejects_legacy_faiss_migration_explicitly(
    tmp_path: Path,
) -> None:
    runtime_root = tmp_path / "runtime"
    runtime_root.mkdir()
    handler = build_daemon_request_handler(
        _router_dependencies(_router_context(tmp_path, "faiss+sqlite"), runtime_root)
    )

    payload = await handler(
        "/api/admin/reindex/submit",
        {"operation": "start", "model": "new-model"},
    )

    assert payload["status"] == "error"
    assert payload["error"] == "reindex_backend_unsupported"
    assert "legacy faiss+sqlite chunk index" in str(payload["details"])


@pytest.mark.asyncio
async def test_daemon_reports_worker_queue_failure_instead_of_accepting_request(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime_root = tmp_path / "runtime"
    runtime_root.mkdir()
    monkeypatch.setattr(
        "mcp_markdown_ragdocs.daemon.request_router.submit_reindex_request",
        lambda *_args, **_kwargs: TaskSubmissionResult(status="unavailable"),
    )
    handler = build_daemon_request_handler(
        _router_dependencies(_router_context(tmp_path, "pgvector"), runtime_root)
    )

    payload = await handler(
        "/api/admin/reindex/submit",
        {"operation": "start", "model": "new-model"},
    )

    assert payload == {
        "status": "error",
        "error": "reindex_queue_unavailable",
        "details": "Index worker is unavailable.",
    }


@pytest.fixture
def reset_task_registration():
    tasks_module._huey = None
    tasks_module._index_manager = None
    tasks_module._bootstrap_index_path = None
    tasks_module.reindex_model_task = None
    yield
    tasks_module._huey = None
    tasks_module._index_manager = None
    tasks_module._bootstrap_index_path = None
    tasks_module.reindex_model_task = None


def test_worker_returns_reindex_failure_and_persists_status(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    reset_task_registration,
) -> None:
    runtime_root = tmp_path / "runtime"
    runtime_root.mkdir()
    config = _config(tmp_path)
    manager = cast(Any, SimpleNamespace(_config=config))
    monkeypatch.setattr(
        tasks_module,
        "run_reindex_operation",
        lambda **_kwargs: (_ for _ in ()).throw(ReindexError("provider failed")),
    )
    huey = SqliteHuey(name="reindex-worker", filename=str(tmp_path / "queue.db"))
    register_tasks(
        huey,
        manager,
        TaskLeaseStore(tmp_path / "queue.db"),
        WorkIntentStore(tmp_path / "queue.db"),
        bootstrap_index_path=runtime_root,
    )

    submission = tasks_module.submit_reindex_request(
        "start",
        model="new-model",
        truncate_dim=1024,
        old_model=None,
        request_id="request-1",
    )
    task = huey.dequeue()
    assert submission.enqueued
    assert task is not None
    result = huey.execute(task)

    assert result == {
        "status": "error",
        "request_id": "request-1",
        "error": "provider failed",
    }
    assert (runtime_root / "reindex-status.json").exists()
    assert (
        reindex_status_payload(runtime_root, tmp_path / "index")["status"]
        == "failed"
    )


def test_cli_surfaces_reindex_failure_to_user(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "mcp_markdown_ragdocs.cli._request_daemon_json",
        lambda *_args, **_kwargs: {
            "status": "error",
            "error": "reindex_backend_unsupported",
            "details": (
                "durable model migration requires store.backend = 'pgvector'; "
                "the legacy faiss+sqlite chunk index is not model-scoped"
            ),
        },
    )

    result = CliRunner().invoke(
        cli,
        ["index", "reindex", "--model", "new-model"],
    )

    assert result.exit_code == 1
    assert "legacy faiss+sqlite chunk index is not model-scoped" in result.output
