"""Bounded, durable Google Drive recovery for failed or stale records."""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from searchkernel.api import Record, atomic_write_json

from mcp_markdown_ragdocs.adapters.sources.gdrive import GoogleDriveContentSource
from mcp_markdown_ragdocs.gdrive.records import (
    extraction_profile,
    processing_fingerprint,
)
from mcp_markdown_ragdocs.gdrive.models import DriveFile, DriveScope

BACKFILL_SCHEMA_VERSION = 1
BACKFILL_CHECKPOINT_FILENAME = "gdrive-backfill-checkpoints.json"
_RETRYABLE_STATUSES = {
    "retryable-error",
    "provider-retryable",
    "provider-transient",
    "provider-rate-limit",
    "provider-authentication",
    "shortcut-unresolved",
}


class DriveBackfillWriter(Protocol):
    def index_records(self, records: Sequence[Record]) -> bool: ...

    def persist(self) -> None: ...


@dataclass(frozen=True, slots=True)
class GDriveBackfillCheckpoint:
    generation: str
    page_token: str | None = None
    batch: int = 0

    def __post_init__(self) -> None:
        if not self.generation:
            raise ValueError("backfill generation must be non-empty")
        if self.page_token is not None and not self.page_token:
            raise ValueError("backfill page_token must be non-empty or null")
        if self.batch < 0:
            raise ValueError("backfill batch must be non-negative")

    def to_payload(self) -> dict[str, object]:
        return {
            "generation": self.generation,
            "page_token": self.page_token,
            "batch": self.batch,
        }

    @classmethod
    def from_payload(cls, payload: object) -> "GDriveBackfillCheckpoint":
        if not isinstance(payload, dict):
            raise ValueError("Google Drive backfill checkpoint must be an object")
        generation = payload.get("generation")
        page_token = payload.get("page_token")
        batch = payload.get("batch")
        if not isinstance(generation, str):
            raise ValueError("backfill generation must be a string")
        if page_token is not None and not isinstance(page_token, str):
            raise ValueError("backfill page_token must be a string or null")
        if not isinstance(batch, int) or isinstance(batch, bool):
            raise ValueError("backfill batch must be an integer")
        return cls(generation, page_token, batch)


class GDriveBackfillCheckpointStore:
    """Atomically persist independent backfill progress per scope."""

    def __init__(self, index_root: Path) -> None:
        self.path = Path(index_root) / BACKFILL_CHECKPOINT_FILENAME

    def load(self, namespace: str, generation: str) -> GDriveBackfillCheckpoint | None:
        checkpoint = self._read().get(namespace)
        if checkpoint is None or checkpoint.generation != generation:
            return None
        return checkpoint

    def begin(self, namespace: str, generation: str) -> GDriveBackfillCheckpoint:
        checkpoint = GDriveBackfillCheckpoint(generation)
        self._save(namespace, checkpoint)
        return checkpoint

    def persist_after_index(
        self,
        namespace: str,
        *,
        generation: str,
        page_token: str | None,
        batch: int,
    ) -> GDriveBackfillCheckpoint:
        current = self.load(namespace, generation)
        if current is None:
            raise ValueError("backfill must begin before progress is persisted")
        if batch != current.batch + 1:
            raise ValueError("backfill batches must advance in order")
        checkpoint = GDriveBackfillCheckpoint(generation, page_token, batch)
        self._save(namespace, checkpoint)
        return checkpoint

    def _read(self) -> dict[str, GDriveBackfillCheckpoint]:
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (FileNotFoundError, OSError, json.JSONDecodeError):
            return {}
        if not isinstance(payload, dict) or payload.get("schema_version") != BACKFILL_SCHEMA_VERSION:
            return {}
        raw_checkpoints = payload.get("checkpoints")
        if not isinstance(raw_checkpoints, dict):
            return {}
        checkpoints: dict[str, GDriveBackfillCheckpoint] = {}
        for namespace, raw in raw_checkpoints.items():
            try:
                checkpoints[str(namespace)] = GDriveBackfillCheckpoint.from_payload(raw)
            except ValueError:
                continue
        return checkpoints

    def _save(self, namespace: str, checkpoint: GDriveBackfillCheckpoint) -> None:
        checkpoints = {
            key: value.to_payload() for key, value in self._read().items()
        }
        checkpoints[namespace] = checkpoint.to_payload()
        atomic_write_json(
            self.path,
            {"schema_version": BACKFILL_SCHEMA_VERSION, "checkpoints": checkpoints},
        )


@dataclass(frozen=True, slots=True)
class GDriveBackfillProgress:
    namespace: str
    generation: str
    pages_scanned: int
    items_scanned: int
    items_reprocessed: int
    complete: bool


class GoogleDriveBackfill:
    """Reprocess a bounded Drive inventory without disturbing sync cursors."""

    def __init__(
        self,
        source: GoogleDriveContentSource,
        checkpoint_store: GDriveBackfillCheckpointStore,
        record_writer: DriveBackfillWriter,
        *,
        scope_generation: str,
        page_size: int = 1000,
        max_items: int = 100_000,
        max_pages: int = 500,
        max_seconds: float = 10.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not scope_generation:
            raise ValueError("scope_generation is required")
        if page_size < 1 or max_items < 1 or max_pages < 1 or max_seconds <= 0:
            raise ValueError("backfill bounds must be positive")
        self.source = source
        self.checkpoint_store = checkpoint_store
        self.record_writer = record_writer
        self.scope_generation = scope_generation
        self.page_size = page_size
        self.max_items = max_items
        self.max_pages = max_pages
        self.max_seconds = max_seconds
        self._clock = clock
        self._scope_source_ids: dict[str, set[str]] = {}

    async def run(
        self,
        scope: DriveScope,
        existing_records: Mapping[str, Record] | Iterable[Record],
    ) -> GDriveBackfillProgress:
        generation = f"{self.source.extractor_version}:{self.source.chunker_version}"
        namespace = f"{self.scope_generation}:{self.source.scope_identity(scope)}"
        records_by_id = (
            {source_id: record for source_id, record in existing_records.items()}
            if isinstance(existing_records, Mapping)
            else {record.source_id: record for record in existing_records}
        )
        checkpoint = self.checkpoint_store.load(namespace, generation)
        if checkpoint is None:
            checkpoint = self.checkpoint_store.begin(namespace, generation)
            self._scope_source_ids[namespace] = set()

        observed_source_ids = self._scope_source_ids.setdefault(namespace, set())
        scope_identity = self.source.scope_identity(scope)

        pages_scanned = 0
        items_scanned = 0
        items_reprocessed = 0
        started_at = self._clock()
        page_token = checkpoint.page_token
        while pages_scanned < self.max_pages and items_scanned < self.max_items:
            if self._clock() - started_at >= self.max_seconds:
                break
            page = await self.source.client.list_files_page(
                scope,
                page_token=page_token,
                page_size=min(self.page_size, self.max_items - items_scanned),
            )
            remaining = self.max_items - items_scanned
            files = page.files[:remaining]
            records: list[Record] = []
            for file in files:
                items_scanned += 1
                existing = records_by_id.get(file.id)
                if self._needs_reprocessing(file, existing):
                    record = await self.source.materialize_record(file, scope=scope)
                    records.append(record)
                    source_id = record.source_id
                else:
                    source_id = existing.source_id if existing is not None else file.id
                    self.source.membership_store.add(
                        self.source.workspace_id,
                        file.id,
                        scope_identity,
                    )
                if scope_identity in self.source.membership_store.memberships_for(
                    self.source.workspace_id, source_id
                ):
                    observed_source_ids.add(source_id)
            complete_backfill = page.next_page_token is None
            reprocessed_count = len(records)
            if complete_backfill:
                if checkpoint.page_token is not None and self.source.membership_store.is_durable:
                    observed_source_ids = set(
                        await self.source.collect_scope_source_ids(scope)
                    )
                    self._scope_source_ids[namespace] = observed_source_ids
                records.extend(
                    self.source.scope_loss_tombstones(scope, observed_source_ids)
                )
            if records:
                if self.record_writer.index_records(records) is False:
                    raise RuntimeError("Google Drive backfill record indexing failed")
                self.record_writer.persist()
                items_reprocessed += reprocessed_count
            if complete_backfill:
                self.source.reconcile_scope(scope, observed_source_ids)
            pages_scanned += 1
            page_token = page.next_page_token
            checkpoint = self.checkpoint_store.persist_after_index(
                namespace,
                generation=generation,
                page_token=page_token,
                batch=checkpoint.batch + 1,
            )
            if page_token is None:
                self._scope_source_ids.pop(namespace, None)
                return GDriveBackfillProgress(
                    namespace,
                    generation,
                    pages_scanned,
                    items_scanned,
                    items_reprocessed,
                    True,
                )

        return GDriveBackfillProgress(
            namespace,
            generation,
            pages_scanned,
            items_scanned,
            items_reprocessed,
            False,
        )

    def _needs_reprocessing(self, file: DriveFile, existing: Record | None) -> bool:
        if existing is None:
            return True
        metadata = existing.metadata
        if metadata.get("deleted") is True or getattr(existing.status, "value", "") == "archived":
            return False
        expected = processing_fingerprint(
            file,
            extraction_profile(file),
            extractor_version=self.source.extractor_version,
            chunker_version=self.source.chunker_version,
        )
        if metadata.get("processing_fingerprint") != expected:
            return True
        return metadata.get("extraction_status") in _RETRYABLE_STATUSES


__all__ = [
    "BACKFILL_CHECKPOINT_FILENAME",
    "BACKFILL_SCHEMA_VERSION",
    "DriveBackfillWriter",
    "GDriveBackfillCheckpoint",
    "GDriveBackfillCheckpointStore",
    "GDriveBackfillProgress",
    "GoogleDriveBackfill",
]
