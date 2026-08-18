"""Progressive bootstrap orchestration over the shared indexing coordinator."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol

from searchkernel.api import (
    CoordinatorReceipt,
    IngestionReceipt,
    Record,
    RecordIngestionResult,
    SearchAvailability,
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

    def embed_query(self, text: str) -> list[float]:
        return self._vector.get_query_embedding(text)


def run_progressive_bootstrap(
    manager: ProgressiveIndexManager,
    file_paths: Sequence[str],
    *,
    documents_roots: Sequence[Path],
) -> CoordinatorReceipt:
    """Run one bounded bootstrap source through the shared coordinator."""
    pending_paths = _pending_canonical_paths(
        manager.index_path,
        file_paths,
        documents_roots,
    )
    receipt = _run_canonical_bootstrap(manager, pending_paths)
    successful_paths = [
        record.source_id
        for record in receipt.ingestion.records
        if record.status == "committed"
    ]
    mark_bootstrap_files_completed(
        manager.index_path,
        list(documents_roots),
        successful_paths,
    )
    if len(successful_paths) == len(pending_paths):
        publish_bootstrap_availability(
            manager.index_path,
            SearchAvailability(
                lexical="available",
                graph="available",
                semantic_coarse="complete",
                semantic_fine="complete",
            ),
        )
    return receipt


def _pending_canonical_paths(
    index_path: Path,
    file_paths: Sequence[str],
    documents_roots: Sequence[Path],
) -> list[str]:
    checkpoint = load_bootstrap_checkpoint(index_path)
    if checkpoint is None:
        return list(file_paths)

    pending_paths: list[str] = []
    for file_path in file_paths:
        relative_path = _relative_path(file_path, documents_roots)
        if (
            relative_path in checkpoint.targets
            and checkpoint.completed.get(relative_path)
            == checkpoint.targets[relative_path]
        ):
            continue
        pending_paths.append(file_path)
    return pending_paths


def _run_canonical_bootstrap(
    manager: Any,
    file_paths: Sequence[str],
) -> CoordinatorReceipt:
    async def run() -> CoordinatorReceipt:
        outcomes: list[RecordIngestionResult] = []
        defer_graph = hasattr(manager, "rebuild_graph")
        for file_path in file_paths:
            try:
                if defer_graph:
                    success = await asyncio.to_thread(
                        manager.index_document,
                        file_path,
                        update_graph=False,
                    )
                else:
                    success = await asyncio.to_thread(
                        manager.index_document,
                        file_path,
                    )
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
        if defer_graph and any(
            outcome.status == "committed" for outcome in outcomes
        ):
            await asyncio.to_thread(manager.rebuild_graph)
        ingestion = IngestionReceipt(
            source_kind="note",
            workspace_id=None,
            checkpoint=None,
            records=tuple(outcomes),
        )
        manager.persist()
        return CoordinatorReceipt(ingestion=ingestion)

    return asyncio.run(run())


def _chunk_for_record(prepared: Any, record: Record) -> Any | None:
    if record.source_kind == "markdown-empty":
        return None
    return next(
        (chunk for chunk in prepared.chunks if chunk.chunk_id == record.source_id),
        None,
    )


def _relative_path(file_path: str, documents_roots: Sequence[Path]) -> str:
    resolved = Path(file_path).resolve()
    for root in documents_roots:
        try:
            return str(resolved.relative_to(root.resolve()))
        except ValueError:
            continue
    raise ValueError(f"file is outside configured document roots: {file_path}")
