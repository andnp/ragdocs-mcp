"""Progressive bootstrap orchestration over the shared indexing coordinator."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol, cast

from searchkernel.api import (
    CoordinatorReceipt,
    IngestionFailureMode,
    IngestionReceipt,
    JsonCheckpointStore,
    Record,
    RecordIngestionResult,
    ResumableSemanticCoordinator,
    SearchAvailability,
    SemanticRecordIngestor,
    SourceBatch,
    get_semantic_completion_status,
    load_bootstrap_checkpoint,
    mark_bootstrap_files_completed,
    mark_semantic_work_completed,
    publish_bootstrap_availability,
)


class ProgressiveIndexManager(Protocol):
    _encoder_fingerprint: Any
    _embedding_cache: Any
    vector: Any
    index_path: Path

    def prepare_progressive_document(self, file_path: str) -> Any: ...

    def apply_progressive_lexical_graph(
        self,
        prepared_documents: Sequence[Any],
    ) -> None: ...

    def finalize_progressive_documents(
        self,
        prepared_documents: Sequence[Any],
    ) -> None: ...

    def persist(self) -> None: ...


class _BootstrapFileSource:
    source_kind = "markdown-bootstrap"

    def __init__(
        self,
        manager: ProgressiveIndexManager,
        file_paths: Sequence[str],
        documents_roots: Sequence[Path],
        prepared_by_file: dict[str, Any],
        records_by_file: dict[str, tuple[Record, ...]],
    ) -> None:
        self._manager = manager
        self._file_paths = tuple(sorted(file_paths))
        self._documents_roots = tuple(documents_roots)
        self.prepared_by_file = prepared_by_file
        self.records_by_file = records_by_file

    async def iter_batches(
        self,
        since: str | None = None,
    ) -> AsyncIterator[SourceBatch]:
        for file_path in self._file_paths:
            relative_path = _relative_path(file_path, self._documents_roots)
            if since is not None and relative_path <= since:
                continue

            prepared = await asyncio.to_thread(
                self._manager.prepare_progressive_document,
                file_path,
            )
            records = _records_for_prepared_document(
                prepared,
                file_path=file_path,
                relative_path=relative_path,
            )
            self.prepared_by_file[relative_path] = prepared
            self.records_by_file[relative_path] = records
            yield SourceBatch(records=records, terminal_cursor=relative_path)


class _VectorEmbeddingEncoder:
    def __init__(self, vector: Any) -> None:
        self._vector = vector

    def encode(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        return [self._vector.get_text_embedding(text) for text in texts]


class _VectorChunkMaterializer:
    def __init__(
        self,
        manager: ProgressiveIndexManager,
        *,
        index_path: Path,
        documents_roots: Sequence[Path],
        target_paths: set[str],
        encoder_namespace: str,
    ) -> None:
        self._manager = manager
        self._index_path = index_path
        self._documents_roots = tuple(documents_roots)
        self._target_paths = target_paths
        self._encoder_namespace = encoder_namespace
        self._chunks: dict[str, Any] = {}
        self._pending_by_file: dict[str, set[str]] = {}
        self._prepared_by_file: dict[str, Any] = {}
        self._file_by_source_id: dict[str, str] = {}

    def register_file(
        self,
        relative_path: str,
        prepared: Any,
        records: Sequence[Record],
    ) -> None:
        self._prepared_by_file[relative_path] = prepared
        pending = self._pending_by_file.setdefault(relative_path, set())
        for record in records:
            source_id = record.storage_key
            pending.add(source_id)
            self._file_by_source_id[source_id] = relative_path
            chunk = _chunk_for_record(prepared, record)
            if chunk is not None:
                self._chunks[source_id] = chunk

    def materialize(
        self,
        source_id: str,
        vector: Sequence[float],
        semantic_input: Any,
    ) -> None:
        relative_path = self._file_by_source_id.get(source_id)
        if relative_path is None:
            raise ValueError(f"semantic input does not belong to the bootstrap: {source_id}")

        chunk = self._chunks.get(source_id)
        if chunk is not None:
            self._manager._embedding_cache.put_many(
                {semantic_input.content_hash: vector}
            )
            self._manager.vector.add_chunk(chunk)

        pending = self._pending_by_file[relative_path]
        pending.discard(source_id)
        if not pending:
            self._complete_file(relative_path)

    def _complete_file(self, relative_path: str) -> None:
        prepared = self._prepared_by_file[relative_path]
        self._manager.finalize_progressive_documents([prepared])
        self._manager.persist()
        mark_semantic_work_completed(
            self._index_path,
            self._encoder_namespace,
            relative_path,
        )
        mark_bootstrap_files_completed(
            self._index_path,
            list(self._documents_roots),
            [prepared.file_path],
        )

        semantic_status = get_semantic_completion_status(
            self._index_path,
            self._encoder_namespace,
        )
        complete = self._target_paths.issubset(
            {path for path, done in semantic_status.items() if done}
        )
        publish_bootstrap_availability(
            self._index_path,
            SearchAvailability(
                lexical="available",
                graph="available",
                semantic_coarse="complete" if complete else "backfilling",
                semantic_fine="complete" if complete else "backfilling",
            ),
        )


class _ProgressiveRecordIngestor:
    def __init__(
        self,
        manager: ProgressiveIndexManager,
        *,
        index_path: Path,
        documents_roots: Sequence[Path],
        prepared_by_file: dict[str, Any],
        records_by_file: dict[str, tuple[Record, ...]],
        materializer: _VectorChunkMaterializer,
    ) -> None:
        self._manager = manager
        self._index_path = index_path
        self._documents_roots = tuple(documents_roots)
        self._prepared_by_file = prepared_by_file
        self._records_by_file = records_by_file
        self._materializer = materializer
        self._staged_files: set[str] = set()

    async def index_records(
        self,
        records: Sequence[Record],
        *,
        checkpoint: str | None = None,
        failure_mode: IngestionFailureMode = "strict",
    ) -> IngestionReceipt:
        return await asyncio.to_thread(
            self._index_records,
            records,
            checkpoint,
            failure_mode,
        )

    def _index_records(
        self,
        records: Sequence[Record],
        checkpoint: str | None,
        failure_mode: IngestionFailureMode,
    ) -> IngestionReceipt:
        try:
            relative_paths = {
                _record_relative_path(record, self._documents_roots)
                for record in records
            }
            for relative_path in relative_paths:
                if relative_path in self._staged_files:
                    continue
                prepared = self._prepared_by_file[relative_path]
                self._manager.apply_progressive_lexical_graph([prepared])
                self._manager.persist()
                self._materializer.register_file(
                    relative_path,
                    prepared,
                    self._records_by_file[relative_path],
                )
                self._staged_files.add(relative_path)
                publish_bootstrap_availability(
                    self._index_path,
                    SearchAvailability(
                        lexical="available",
                        graph="available",
                        semantic_coarse="backfilling",
                        semantic_fine="backfilling",
                    ),
                )
        except Exception:
            if failure_mode == "strict":
                raise
            return IngestionReceipt(
                source_kind="markdown-bootstrap",
                workspace_id=None,
                checkpoint=checkpoint,
                records=tuple(
                    RecordIngestionResult(
                        source_kind=record.source_kind,
                        source_id=record.source_id,
                        workspace_id=record.workspace_id,
                        status="failed",
                        error="lexical or graph indexing failed",
                    )
                    for record in records
                ),
            )

        return IngestionReceipt(
            source_kind="markdown-bootstrap",
            workspace_id=None,
            checkpoint=checkpoint,
            records=tuple(
                RecordIngestionResult(
                    source_kind=record.source_kind,
                    source_id=record.source_id,
                    workspace_id=record.workspace_id,
                    status="committed",
                )
                for record in records
            ),
        )


class _NoopKeywordStore:
    def index(self, records: list[Record]) -> None:
        _ = records
        return


class _NoopVectorStore:
    def upsert(
        self,
        records: list[Record],
        model_name: str,
        dim: int,
    ) -> None:
        _ = records, model_name, dim
        return


class _VectorEmbeddingProvider:
    def __init__(self, vector: Any, model_name: str) -> None:
        self._vector = vector
        self.model_name = model_name
        self.dim = 0

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._vector.get_text_embedding(text) for text in texts]


def run_progressive_bootstrap(
    manager: ProgressiveIndexManager,
    file_paths: Sequence[str],
    *,
    documents_roots: Sequence[Path],
) -> CoordinatorReceipt:
    """Run one bounded bootstrap source through the shared coordinator."""
    if hasattr(manager, "kernel"):
        return _run_canonical_bootstrap(manager, file_paths)

    target_paths = {
        _relative_path(file_path, documents_roots)
        for file_path in file_paths
    }
    checkpoint = load_bootstrap_checkpoint(manager.index_path)
    if checkpoint is not None:
        target_paths = set(checkpoint.targets)
    encoder_namespace = manager._encoder_fingerprint.namespace
    prepared_by_file: dict[str, Any] = {}
    records_by_file: dict[str, tuple[Record, ...]] = {}
    source = _BootstrapFileSource(
        manager,
        file_paths,
        documents_roots,
        prepared_by_file,
        records_by_file,
    )
    materializer = _VectorChunkMaterializer(
        manager,
        index_path=manager.index_path,
        documents_roots=documents_roots,
        target_paths=target_paths,
        encoder_namespace=encoder_namespace,
    )
    ingestor = _ProgressiveRecordIngestor(
        manager,
        index_path=manager.index_path,
        documents_roots=documents_roots,
        prepared_by_file=prepared_by_file,
        records_by_file=records_by_file,
        materializer=materializer,
    )
    planner_source = SemanticRecordIngestor(
        embedding_provider=cast(
            Any,
            _VectorEmbeddingProvider(
                manager.vector,
                manager._encoder_fingerprint.model,
            ),
        ),
        keyword_store=cast(Any, _NoopKeywordStore()),
        vector_store=cast(Any, _NoopVectorStore()),
        embedding_cache=manager._embedding_cache,
        encoder_namespace=encoder_namespace,
    )
    planner = getattr(planner_source, "_planner")
    coordinator = ResumableSemanticCoordinator(
        planner=planner,
        cache=manager._embedding_cache,
        encoder=_VectorEmbeddingEncoder(manager.vector),
        materializer=materializer,
        record_ingestor=ingestor,
        checkpoint_store=JsonCheckpointStore(
            manager.index_path / "semantic.checkpoint.json"
        ),
    )
    return asyncio.run(
        coordinator.run_source(
            source,
            workspace_id=str(manager.index_path),
            batch_size=64,
            failure_mode="strict",
        )
    )


def _run_canonical_bootstrap(
    manager: Any,
    file_paths: Sequence[str],
) -> CoordinatorReceipt:
    async def run() -> CoordinatorReceipt:
        outcomes: list[RecordIngestionResult] = []
        for file_path in file_paths:
            try:
                success = await asyncio.to_thread(manager.index_document, file_path)
            except Exception as error:  # noqa: BLE001 - worker boundary
                success = False
                error_text = str(error)
            else:
                error_text = None if success else "record indexing failed"
            outcomes.append(
                RecordIngestionResult(
                    source_kind="note",
                    source_id=str(file_path),
                    workspace_id=None,
                    status="committed" if success else "failed",
                    error=error_text,
                )
            )
        ingestion = IngestionReceipt(
            source_kind="note",
            workspace_id=None,
            checkpoint=None,
            records=tuple(outcomes),
        )
        manager.persist()
        return CoordinatorReceipt(ingestion=ingestion)

    return asyncio.run(run())


def _records_for_prepared_document(
    prepared: Any,
    *,
    file_path: str,
    relative_path: str,
) -> tuple[Record, ...]:
    records = tuple(
        Record(
            source_kind="markdown-chunk",
            source_id=chunk.chunk_id,
            title=_chunk_header_path(chunk) or chunk.chunk_id,
            body=(
                f"{_chunk_header_path(chunk)}\n\n{chunk.content}"
                if _chunk_header_path(chunk)
                else chunk.content
            ),
            created_at=_chunk_modified_time(chunk),
            updated_at=_chunk_modified_time(chunk),
            metadata={
                **chunk.metadata,
                "bootstrap_file_path": file_path,
                "bootstrap_relative_path": relative_path,
            },
            uri=str(chunk.metadata.get("file_path", file_path)),
        )
        for chunk in prepared.chunks
    )
    if records:
        return records
    return (
        Record(
            source_kind="markdown-empty",
            source_id=f"empty:{relative_path}",
            title=relative_path,
            body="",
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
            metadata={
                "bootstrap_file_path": file_path,
                "bootstrap_relative_path": relative_path,
            },
            uri=file_path,
        ),
    )


def _chunk_header_path(chunk: Any) -> str:
    header_path = getattr(chunk, "header_path", None)
    if isinstance(header_path, str):
        return header_path
    metadata = getattr(chunk, "metadata", {})
    value = metadata.get("header_path")
    return value if isinstance(value, str) else ""


def _chunk_modified_time(chunk: Any) -> datetime:
    modified_time = getattr(chunk, "modified_time", None)
    if isinstance(modified_time, datetime):
        return modified_time
    metadata = getattr(chunk, "metadata", {})
    value = metadata.get("modified_time")
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        return datetime.fromisoformat(value)
    return datetime.now(UTC)


def _chunk_for_record(prepared: Any, record: Record) -> Any | None:
    if record.source_kind == "markdown-empty":
        return None
    return next(
        (chunk for chunk in prepared.chunks if chunk.chunk_id == record.source_id),
        None,
    )


def _record_relative_path(
    record: Record,
    documents_roots: Sequence[Path],
) -> str:
    relative = record.metadata.get("bootstrap_relative_path")
    if isinstance(relative, str):
        return relative
    raise ValueError(f"bootstrap record has no relative path: {record.source_id}")


def _relative_path(file_path: str, documents_roots: Sequence[Path]) -> str:
    resolved = Path(file_path).resolve()
    for root in documents_roots:
        try:
            return str(resolved.relative_to(root.resolve()))
        except ValueError:
            continue
    raise ValueError(f"file is outside configured document roots: {file_path}")
