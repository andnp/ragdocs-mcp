"""Application orchestration for durable embedding-model migrations."""

from __future__ import annotations

import hashlib
import json
import re
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

import psycopg2
from psycopg2 import sql
from psycopg2.extras import Json

from searchkernel.api import (
    ActiveModelMetadata,
    BackupMetadata,
    CURRENT_MANIFEST_SPEC_VERSION,
    EmbeddingProvider,
    IndexManifest,
    MigrationState,
    ModelDimensionMismatchError,
    ModelLifecycleStore,
    ModelNamespace,
    ModelNamespaceStore,
    Record,
    RecordStatus,
    ReindexError,
    ReindexRoutine,
    RollbackMetadata,
    ValidationResult,
    VectorStore,
    atomic_write_json,
    load_manifest,
    save_manifest,
)

from mcp_markdown_ragdocs.config import Config, resolve_embedding_model
from mcp_markdown_ragdocs.coordination.file_lock import IndexLock

REINDEX_ACTIVE_STATUSES = {"queued", "running"}
REINDEX_TERMINAL_STATUSES = {"succeeded", "failed"}
_MODEL_NAME_RE = re.compile(r"[^a-z0-9_]+")


def default_reindex_status() -> dict[str, object]:
    return {
        "status": "idle",
        "operation": None,
        "request_id": None,
        "model": None,
        "truncate_dim": None,
        "old_model": None,
        "phase": "idle",
        "checkpoint": 0,
        "total_records": 0,
        "error": None,
        "submitted_at": None,
        "started_at": None,
        "completed_at": None,
    }


def reindex_status_path(runtime_root: Path) -> Path:
    return runtime_root / "reindex-status.json"


def read_reindex_status(runtime_root: Path) -> dict[str, object]:
    path = reindex_status_path(runtime_root)
    if not path.exists():
        return default_reindex_status()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default_reindex_status()
    if not isinstance(payload, dict):
        return default_reindex_status()
    return {**default_reindex_status(), **payload}


def write_reindex_status(
    runtime_root: Path,
    payload: dict[str, object],
) -> dict[str, object]:
    normalized = {**default_reindex_status(), **payload}
    atomic_write_json(reindex_status_path(runtime_root), normalized)
    return normalized


def submit_reindex_status(
    runtime_root: Path,
    *,
    operation: str,
    request_id: str,
    model: str | None,
    truncate_dim: int | None,
    old_model: str | None,
) -> dict[str, object]:
    return write_reindex_status(
        runtime_root,
        {
            "status": "queued",
            "operation": operation,
            "request_id": request_id,
            "model": model,
            "truncate_dim": truncate_dim,
            "old_model": old_model,
            "phase": "queued",
            "checkpoint": 0,
            "total_records": 0,
            "error": None,
            "submitted_at": datetime.now(UTC).isoformat(),
            "started_at": None,
            "completed_at": None,
        },
    )


def reindex_status_payload(
    runtime_root: Path,
    index_path: Path,
) -> dict[str, object]:
    payload = read_reindex_status(runtime_root)
    manifest = load_manifest(index_path)
    if manifest is not None:
        payload["active_model"] = (
            manifest.active_model.to_dict()
            if manifest.active_model is not None
            else None
        )
        payload["migration"] = (
            manifest.migration.to_dict() if manifest.migration is not None else None
        )
        if manifest.migration is not None:
            payload["phase"] = manifest.migration.phase.value
            payload["checkpoint"] = manifest.migration.checkpoint
            payload["total_records"] = manifest.migration.total_records or 0
            if manifest.migration.error is not None:
                payload["error"] = manifest.migration.error
            if manifest.migration.phase.value == "complete":
                payload["status"] = "succeeded"
            elif manifest.migration.phase.value == "failed":
                payload["status"] = "failed"
    return payload


class _HuggingFaceEmbeddingProvider:
    """Adapt the supported local provider to the public kernel contract."""

    def __init__(self, model_name: str, truncate_dim: int | None = None) -> None:
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding

        kwargs: Any = {"model_name": model_name}
        if truncate_dim is not None:
            kwargs["truncate_dim"] = truncate_dim
        self._embedder = HuggingFaceEmbedding(**kwargs)
        self.model_name = model_name
        native_dim = getattr(self._embedder, "embed_dim", None)
        if not isinstance(native_dim, int) or native_dim < 1:
            sample = self._embedder.get_text_embedding("dimension probe")
            native_dim = len(sample)
        self.dim = truncate_dim or native_dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [
            [float(value) for value in self._embedder.get_text_embedding(text)]
            for text in texts
        ]


class _DeferredEmbeddingProvider:
    """Carry model identity for lifecycle-only operations without loading a model."""

    def __init__(
        self,
        model_name: str,
        dim: int,
        factory: Callable[[], EmbeddingProvider],
    ) -> None:
        self.model_name = model_name
        self.dim = dim
        self._factory = factory

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self._factory().embed(texts)


def build_embedding_provider(
    config: Config,
    model_name: str,
    truncate_dim: int | None,
) -> EmbeddingProvider:
    provider_name = config.embedding.provider.lower()
    if provider_name not in {"hf", "huggingface", "local"}:
        raise ReindexError(
            f"unsupported reindex embedding provider {config.embedding.provider!r}"
        )
    resolved_model = (
        resolve_embedding_model(config) if model_name == "local" else model_name
    )
    return _HuggingFaceEmbeddingProvider(resolved_model, truncate_dim)


class ManifestModelLifecycleStore(ModelLifecycleStore):
    """Persist migration state and active-model metadata in the app manifest."""

    def __init__(
        self,
        index_path: Path,
        *,
        source_namespace: ModelNamespace,
        manifest_template: IndexManifest | None = None,
        namespace_store: Any | None = None,
    ) -> None:
        self.index_path = index_path
        self.source_namespace = source_namespace
        self.manifest_template = manifest_template
        self.namespace_store = namespace_store

    def _manifest(self) -> IndexManifest:
        manifest = load_manifest(self.index_path)
        if manifest is not None:
            return manifest
        if self.manifest_template is not None:
            return self.manifest_template
        return IndexManifest(
            spec_version=CURRENT_MANIFEST_SPEC_VERSION,
            embedding_model=self.source_namespace.model_name,
            chunking_config={},
            indexed_files={},
        )

    def ensure_namespace(self, namespace: ModelNamespace) -> None:
        if self.namespace_store is None:
            raise ReindexError("model namespace storage is not configured")
        self.namespace_store.ensure_namespace(namespace)

    def delete_namespace(self, namespace: ModelNamespace) -> None:
        if self.namespace_store is None:
            raise ReindexError("model namespace storage is not configured")
        self.namespace_store.delete_namespace(namespace)

    def validate_namespace(
        self,
        namespace: ModelNamespace,
        expected_records: int,
    ) -> ValidationResult:
        if self.namespace_store is None:
            raise ReindexError("model namespace storage is not configured")
        return self.namespace_store.validate_namespace(namespace, expected_records)

    def get_active_model(self) -> ActiveModelMetadata | None:
        manifest = load_manifest(self.index_path)
        if manifest is not None and manifest.active_model is not None:
            return manifest.active_model
        return ActiveModelMetadata(namespace=self.source_namespace)

    def set_active_model(self, active_model: ActiveModelMetadata) -> None:
        manifest = self._manifest()
        manifest.active_model = active_model
        manifest.embedding_model = active_model.namespace.model_name
        save_manifest(self.index_path, manifest)

    def compare_and_set_active_model(
        self,
        expected: ActiveModelMetadata | None,
        active_model: ActiveModelMetadata,
    ) -> bool:
        if self.get_active_model() != expected:
            return False
        self.set_active_model(active_model)
        return True

    def create_backup(self, namespace: ModelNamespace) -> BackupMetadata:
        if self.namespace_store is None:
            raise ReindexError("model namespace storage is not configured")
        return self.namespace_store.create_backup(namespace)

    def restore_backup(self, backup: BackupMetadata) -> RollbackMetadata:
        if self.namespace_store is None:
            raise ReindexError("model namespace storage is not configured")
        return self.namespace_store.restore_backup(backup)

    @contextmanager
    def acquire_transition_lock(self) -> Iterator[None]:
        lock = IndexLock(self.index_path, timeout_seconds=30.0)
        lock.acquire_exclusive()
        try:
            yield None
        finally:
            lock.release()

    def load_migration(self, migration_id: str) -> MigrationState | None:
        manifest = load_manifest(self.index_path)
        if manifest is None or manifest.migration is None:
            return None
        if manifest.migration.migration_id != migration_id:
            return None
        return manifest.migration

    def save_migration(self, migration: MigrationState) -> None:
        manifest = self._manifest()
        manifest.migration = migration
        save_manifest(self.index_path, manifest)


class PgvectorReindexStore(
    VectorStore,
    ModelNamespaceStore,
):
    """App-owned pgvector migration seam using the public Record contract."""

    def __init__(self, pg_dsn: str) -> None:
        if not pg_dsn:
            raise ReindexError(
                "pgvector reindex requires store.pg_dsn or SEARCHKERNEL_PG_DSN"
            )
        self.pg_dsn = pg_dsn

    @contextmanager
    def _connection(self) -> Iterator[Any]:
        connection = psycopg2.connect(self.pg_dsn)
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    @staticmethod
    def _sanitize_model_name(model_name: str) -> str:
        sanitized = _MODEL_NAME_RE.sub("_", model_name.lower()).strip("_") or "model"
        digest = hashlib.sha256(model_name.encode("utf-8")).hexdigest()[:8]
        return f"{sanitized}_{digest}"

    @classmethod
    def _table_name(cls, namespace: ModelNamespace) -> str:
        return f"vectors__{cls._sanitize_model_name(namespace.model_name)}__{namespace.dim}"

    @staticmethod
    def _vector_literal(values: list[float]) -> str:
        return "[" + ",".join(repr(float(value)) for value in values) + "]"

    def _registered_dimensions(self, cursor: Any, model_name: str) -> list[int]:
        cursor.execute(
            "SELECT dim FROM vector_tables WHERE model_name = %s ORDER BY dim;",
            (model_name,),
        )
        return [int(row[0]) for row in cursor.fetchall()]

    def resolve_namespace(self, model_name: str) -> ModelNamespace:
        with self._connection() as connection:
            cursor = connection.cursor()
            dimensions = self._registered_dimensions(cursor, model_name)
            if not dimensions:
                raise ReindexError(
                    f"no pgvector namespace exists for model {model_name!r}"
                )
            if len(set(dimensions)) != 1:
                raise ModelDimensionMismatchError(
                    f"model {model_name!r} has mixed dimensions: {dimensions}"
                )
            return ModelNamespace(model_name, dimensions[0])

    def ensure_namespace(self, namespace: ModelNamespace) -> None:
        with self._connection() as connection:
            cursor = connection.cursor()
            dimensions = self._registered_dimensions(cursor, namespace.model_name)
            if dimensions and any(dim != namespace.dim for dim in dimensions):
                raise ModelDimensionMismatchError(
                    f"Dimension mismatch for model {namespace.model_name}: "
                    f"expected {dimensions[0]}, got {namespace.dim}"
                )
            table_name = self._table_name(namespace)
            cursor.execute(
                sql.SQL(
                    "CREATE TABLE IF NOT EXISTS {table} ("
                    "record_id TEXT PRIMARY KEY, "
                    "embedding vector({dim}) NOT NULL, "
                    "created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP, "
                    "updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP"
                    ");"
                ).format(
                    table=sql.Identifier(table_name),
                    dim=sql.SQL(str(namespace.dim)),
                )
            )
            cursor.execute(
                sql.SQL(
                    "CREATE INDEX IF NOT EXISTS {index_name} ON {table} "
                    "USING hnsw (embedding vector_cosine_ops);"
                ).format(
                    index_name=sql.Identifier(f"idx_{table_name}_hnsw"),
                    table=sql.Identifier(table_name),
                )
            )
            cursor.execute(
                "INSERT INTO vector_tables (model_name, dim, table_name) "
                "VALUES (%s, %s, %s) ON CONFLICT (model_name, dim) DO NOTHING;",
                (namespace.model_name, namespace.dim, table_name),
            )

    def upsert(self, records: list[Record], model_name: str, dim: int) -> None:
        namespace = ModelNamespace(model_name, dim)
        self.ensure_namespace(namespace)
        table_name = self._table_name(namespace)
        with self._connection() as connection:
            cursor = connection.cursor()
            for record in records:
                if record.embedding is None or len(record.embedding) != dim:
                    raise ReindexError(
                        f"record {record.storage_key} has no {dim}-dimensional embedding"
                    )
                indexed_text = record.indexed_text or record.body
                cursor.execute(
                    """
                    INSERT INTO records
                    (record_id, workspace_id, source_kind, source_id, title, body,
                     indexed_text, tsvector_body, created_at, updated_at, metadata, uri, status)
                    VALUES (%s, %s, %s, %s, %s, %s, %s,
                            to_tsvector('english', %s), %s, %s, %s, %s, %s)
                    ON CONFLICT (record_id) DO UPDATE SET
                        workspace_id = EXCLUDED.workspace_id,
                        source_kind = EXCLUDED.source_kind,
                        source_id = EXCLUDED.source_id,
                        title = EXCLUDED.title,
                        body = EXCLUDED.body,
                        indexed_text = EXCLUDED.indexed_text,
                        tsvector_body = EXCLUDED.tsvector_body,
                        created_at = EXCLUDED.created_at,
                        updated_at = EXCLUDED.updated_at,
                        metadata = EXCLUDED.metadata,
                        uri = EXCLUDED.uri,
                        status = EXCLUDED.status;
                    """,
                    (
                        record.storage_key,
                        record.workspace_id,
                        record.source_kind,
                        record.source_id,
                        record.title,
                        record.body,
                        record.indexed_text,
                        f"{record.title} {indexed_text}",
                        record.created_at,
                        record.updated_at,
                        Json(record.metadata),
                        record.uri,
                        record.status.value,
                    ),
                )
                cursor.execute(
                    sql.SQL(
                        "INSERT INTO {table} (record_id, embedding) VALUES (%s, %s::vector) "
                        "ON CONFLICT (record_id) DO UPDATE SET "
                        "embedding = EXCLUDED.embedding, updated_at = CURRENT_TIMESTAMP;"
                    ).format(table=sql.Identifier(table_name)),
                    (record.storage_key, self._vector_literal(record.embedding)),
                )

    def search(
        self,
        query_vector: list[float],
        k: int,
        *,
        model_name: str,
        dim: int,
        filters: dict[str, Any] | None = None,
    ) -> list[tuple[str, float]]:
        raise ReindexError("reindex store does not serve queries")

    def delete(self, record_ids: list[str]) -> None:
        raise ReindexError("reindex store requires model-scoped deletion")

    def epoch(self) -> int:
        return 0

    def validate_namespace(
        self,
        namespace: ModelNamespace,
        expected_records: int,
    ) -> ValidationResult:
        table_name = self._table_name(namespace)
        with self._connection() as connection:
            cursor = connection.cursor()
            cursor.execute(
                sql.SQL(
                    "SELECT COUNT(*) FROM {table} v "
                    "JOIN records r ON r.record_id = v.record_id "
                    "WHERE r.source_kind = 'chunk';"
                ).format(table=sql.Identifier(table_name))
            )
            indexed_records = int(cursor.fetchone()[0])
        return ValidationResult(
            namespace=namespace,
            expected_records=expected_records,
            indexed_records=indexed_records,
            checked_at=datetime.now(UTC).isoformat(),
        )

    def create_backup(self, namespace: ModelNamespace) -> BackupMetadata:
        source_table = self._table_name(namespace)
        backup_table = f"reindex_backup_{uuid.uuid4().hex}"
        with self._connection() as connection:
            cursor = connection.cursor()
            cursor.execute(
                sql.SQL(
                    "CREATE TABLE {backup} AS SELECT * FROM {source};"
                ).format(
                    backup=sql.Identifier(backup_table),
                    source=sql.Identifier(source_table),
                )
            )
        return BackupMetadata(
            backup_id=backup_table,
            namespace=namespace,
            reference=backup_table,
            created_at=datetime.now(UTC).isoformat(),
        )

    def restore_backup(self, backup: BackupMetadata) -> RollbackMetadata:
        self.ensure_namespace(backup.namespace)
        target_table = self._table_name(backup.namespace)
        with self._connection() as connection:
            cursor = connection.cursor()
            cursor.execute(
                sql.SQL(
                    "INSERT INTO {target} (record_id, embedding, created_at, updated_at) "
                    "SELECT record_id, embedding, created_at, updated_at FROM {backup} "
                    "ON CONFLICT (record_id) DO UPDATE SET "
                    "embedding = EXCLUDED.embedding, updated_at = EXCLUDED.updated_at;"
                ).format(
                    target=sql.Identifier(target_table),
                    backup=sql.Identifier(backup.reference),
                )
            )
        return RollbackMetadata(
            from_namespace=backup.namespace,
            to_namespace=backup.namespace,
            backup_id=backup.backup_id,
            reason="restored model backup",
            rolled_back_at=datetime.now(UTC).isoformat(),
        )

    def delete_namespace(self, namespace: ModelNamespace) -> None:
        table_name = self._table_name(namespace)
        with self._connection() as connection:
            cursor = connection.cursor()
            cursor.execute(
                sql.SQL("DROP TABLE IF EXISTS {table};").format(
                    table=sql.Identifier(table_name)
                )
            )
            cursor.execute(
                "DELETE FROM vector_tables WHERE model_name = %s AND dim = %s;",
                (namespace.model_name, namespace.dim),
            )

    def load_records(self, namespace: ModelNamespace) -> list[Record]:
        table_name = self._table_name(namespace)
        with self._connection() as connection:
            cursor = connection.cursor()
            cursor.execute(
                sql.SQL(
                    "SELECT r.workspace_id, r.source_kind, r.source_id, r.title, r.body, "
                    "r.created_at, r.updated_at, r.metadata, r.uri, r.status, r.indexed_text "
                    "FROM {table} v JOIN records r ON r.record_id = v.record_id "
                    "WHERE r.source_kind = 'chunk' ORDER BY v.record_id;"
                ).format(table=sql.Identifier(table_name))
            )
            rows = cursor.fetchall()
        records: list[Record] = []
        for (
            workspace_id,
            source_kind,
            source_id,
            title,
            body,
            created_at,
            updated_at,
            metadata,
            uri,
            status,
            indexed_text,
        ) in rows:
            raw_metadata = metadata if isinstance(metadata, dict) else json.loads(metadata)
            records.append(
                Record(
                    workspace_id=workspace_id,
                    source_kind=source_kind,
                    source_id=source_id,
                    title=title,
                    body=body,
                    created_at=created_at,
                    updated_at=updated_at,
                    metadata=raw_metadata,
                    uri=uri,
                    status=RecordStatus(status),
                    indexed_text=indexed_text,
                )
            )
        return records


def _source_namespace(
    config: Config,
    index_path: Path,
    store: PgvectorReindexStore,
) -> ModelNamespace:
    manifest = load_manifest(index_path)
    if manifest is not None and manifest.active_model is not None:
        return manifest.active_model.namespace
    return store.resolve_namespace(resolve_embedding_model(config))


def _manifest_template(config: Config, index_path: Path) -> IndexManifest:
    existing = load_manifest(index_path)
    if existing is not None:
        return existing
    return IndexManifest(
        spec_version=CURRENT_MANIFEST_SPEC_VERSION,
        embedding_model=resolve_embedding_model(config),
        chunking_config={
            "strategy": config.chunking.strategy,
            "min_chunk_chars": config.chunking.min_chunk_chars,
            "max_chunk_chars": config.chunking.max_chunk_chars,
            "overlap_chars": config.chunking.overlap_chars,
        },
        indexed_files={},
    )


def run_reindex_operation(
    *,
    config: Config,
    index_path: Path,
    runtime_root: Path,
    operation: str,
    model: str | None,
    truncate_dim: int | None,
    old_model: str | None,
) -> MigrationState:
    if config.store.backend != "pgvector":
        raise ReindexError(
            "durable model migration is unavailable for the legacy "
            "faiss+sqlite chunk index; configure store.backend = 'pgvector'"
        )

    store = PgvectorReindexStore(config.store.pg_dsn)
    manifest = load_manifest(index_path)
    saved_migration = manifest.migration if manifest is not None else None
    if operation == "start":
        if not model:
            raise ReindexError("a target model is required to start reindex")
        provider = build_embedding_provider(config, model, truncate_dim)
        source = _source_namespace(config, index_path, store)
        records = store.load_records(source)
        target = ModelNamespace(provider.model_name, provider.dim)
        migration_id = f"reindex:{target.model_name}:{target.dim}"
    elif saved_migration is None:
        raise ReindexError("no durable model migration is available")
    else:
        source = saved_migration.source
        target = saved_migration.target
        records = store.load_records(source)
        if not records:
            records = store.load_records(target)
        migration_id = saved_migration.migration_id
        provider = _DeferredEmbeddingProvider(
            target.model_name,
            target.dim,
            lambda: build_embedding_provider(config, target.model_name, target.dim),
        )

    lifecycle = ManifestModelLifecycleStore(
        index_path,
        source_namespace=source,
        manifest_template=_manifest_template(config, index_path),
        namespace_store=store,
    )
    routine = ReindexRoutine(
        records,
        provider,
        store,
        batch_size=max(1, config.embedding.batch_size),
        truncate_dim=target.dim,
        lifecycle_store=lifecycle,
        migration_id=migration_id,
        source_namespace=source,
    )
    if operation == "start":
        state = routine.run()
    elif operation == "contract":
        state = routine.contract(old_model)
    elif operation == "rollback":
        state = routine.rollback(old_model)
    else:
        raise ReindexError(f"unknown reindex operation {operation!r}")

    write_reindex_status(
        runtime_root,
        {
            "status": "succeeded",
            "operation": operation,
            "model": target.model_name,
            "truncate_dim": target.dim,
            "old_model": source.model_name,
            "phase": state.phase.value,
            "checkpoint": state.checkpoint,
            "total_records": state.total_records or len(records),
            "error": None,
            "completed_at": datetime.now(UTC).isoformat(),
        },
    )
    return state
