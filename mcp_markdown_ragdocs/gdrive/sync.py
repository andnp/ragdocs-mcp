"""Resumable Google Drive synchronization flows."""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol

from searchkernel.api import Record

from mcp_markdown_ragdocs.adapters.sources.gdrive import GoogleDriveContentSource
from mcp_markdown_ragdocs.gdrive.checkpoints import (
    GDriveSyncCheckpointStore,
    checkpoint_namespace,
)
from mcp_markdown_ragdocs.gdrive.errors import classify_provider_error
from mcp_markdown_ragdocs.gdrive.models import DriveChange, DriveScope


class DriveRecordWriter(Protocol):
    """Storage boundary used by Drive sync mutations."""

    def index_records(self, records: Sequence[Record]) -> bool: ...

    def persist(self) -> None: ...


@dataclass(frozen=True, slots=True)
class GDriveSyncProgress:
    """Observable progress from one bounded synchronization pass."""

    namespace: str
    start_token: str
    pages_indexed: int
    items_indexed: int
    complete: bool
    token_reset: bool = False


class GoogleDriveSync:
    """Run bounded, checkpointed Drive inventory and change synchronization."""

    def __init__(
        self,
        source: GoogleDriveContentSource,
        checkpoint_store: GDriveSyncCheckpointStore,
        record_writer: DriveRecordWriter,
        *,
        scope_generation: str,
        page_size: int = 1000,
        batch_size: int = 100,
        max_items: int = 100_000,
        max_pages: int = 500,
        max_seconds: float = 10.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not scope_generation or ":" in scope_generation:
            raise ValueError("scope_generation must be non-empty and must not contain ':'")
        if page_size < 1:
            raise ValueError("page_size must be positive")
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if max_items < 1:
            raise ValueError("max_items must be positive")
        if max_pages < 1:
            raise ValueError("max_pages must be positive")
        if max_seconds <= 0:
            raise ValueError("max_seconds must be positive")
        self.source = source
        self.checkpoint_store = checkpoint_store
        self.record_writer = record_writer
        self.scope_generation = scope_generation
        self.page_size = page_size
        self.batch_size = batch_size
        self.max_items = max_items
        self.max_pages = max_pages
        self.max_seconds = max_seconds
        self._clock = clock
        self._inventory_source_ids: dict[str, set[str]] = {}

    async def sync_inventory(self, scope: DriveScope) -> GDriveSyncProgress:
        """Index bounded inventory pages and advance after each durable write."""
        namespace = self._namespace(scope)
        checkpoint = self.checkpoint_store.load(namespace)
        if checkpoint is None or checkpoint.inventory_start_token is None:
            start_token = (await self.source.client.get_start_page_token(scope)).token
            checkpoint = self.checkpoint_store.begin_inventory(namespace, start_token)
            self._inventory_source_ids[namespace] = set()
        else:
            start_token = checkpoint.inventory_start_token

        observed_source_ids = self._inventory_source_ids.setdefault(namespace, set())

        if checkpoint.inventory_batch and checkpoint.inventory_page_token is None:
            return GDriveSyncProgress(namespace, start_token, 0, 0, True)

        resumed_inventory = checkpoint.inventory_page_token is not None
        page_token = checkpoint.inventory_page_token
        pages_indexed = 0
        items_indexed = 0
        started_at = self._clock()
        while pages_indexed < self.max_pages and items_indexed < self.max_items:
            if self._clock() - started_at >= self.max_seconds:
                break
            page = await self.source.client.list_files_page(
                scope,
                page_token=page_token,
                page_size=min(self.page_size, self.max_items - items_indexed),
            )
            records = [
                await self.source.materialize_record(file, scope=scope)
                for file in page.files
            ]
            observed_source_ids.update(
                record.source_id
                for record in records
                if self._record_belongs_to_scope(
                    record, self.source.scope_identity(scope)
                    )
            )
            complete_inventory = page.next_page_token is None
            if complete_inventory:
                if resumed_inventory and self.source.membership_store.is_durable:
                    observed_source_ids = set(
                        await self.source.collect_scope_source_ids(scope)
                    )
                    self._inventory_source_ids[namespace] = observed_source_ids
                records.extend(
                    self.source.scope_loss_tombstones(scope, observed_source_ids)
                )
            self._index_and_persist(records, "inventory")
            if complete_inventory:
                self.source.reconcile_scope(scope, observed_source_ids)
            pages_indexed += 1
            items_indexed += len(records)
            checkpoint = self.checkpoint_store.persist_inventory_batch_after_index(
                namespace,
                page_token=page.next_page_token,
                batch=checkpoint.inventory_batch + 1,
            )
            page_token = page.next_page_token
            if page_token is None:
                self._inventory_source_ids.pop(namespace, None)
                return GDriveSyncProgress(
                    namespace,
                    start_token,
                    pages_indexed,
                    items_indexed,
                    True,
                )

        return GDriveSyncProgress(
            namespace,
            start_token,
            pages_indexed,
            items_indexed,
            False,
        )

    @staticmethod
    def _record_belongs_to_scope(record: Record, scope_identity: str) -> bool:
        memberships = record.metadata.get("scope_memberships")
        return isinstance(memberships, (list, tuple)) and scope_identity in memberships

    async def sync_changes(self, scope: DriveScope) -> GDriveSyncProgress:
        """Replay bounded change pages after inventory has completed."""
        namespace = self._namespace(scope)
        checkpoint = self.checkpoint_store.load(namespace)
        if checkpoint is None or checkpoint.inventory_start_token is None:
            raise ValueError("Google Drive changes require an inventory checkpoint")
        if checkpoint.inventory_batch == 0 or checkpoint.inventory_page_token is not None:
            raise ValueError("Google Drive changes require completed inventory")

        cursor = checkpoint.changes_token or checkpoint.inventory_start_token
        pages_indexed = 0
        items_indexed = 0
        started_at = self._clock()
        while pages_indexed < self.max_pages and items_indexed < self.max_items:
            if self._clock() - started_at >= self.max_seconds:
                break
            try:
                page = await self.source.client.list_changes_page(
                    scope,
                    cursor,
                    page_size=min(self.page_size, self.max_items - items_indexed),
                )
            except Exception as error:
                if is_invalidated_change_token(error):
                    return await self.resync_after_token_reset(scope)
                raise
            records = await self._materialize_changes(page.changes, scope)
            self._index_and_persist(records, "change")
            pages_indexed += 1
            items_indexed += len(records)
            next_cursor = page.next_page_token or page.new_start_page_token
            if not next_cursor:
                raise ValueError("Google Drive change page did not provide a cursor")
            checkpoint = self.checkpoint_store.persist_changes_after_index(
                namespace,
                next_cursor,
            )
            cursor = next_cursor
            if page.next_page_token is None:
                return GDriveSyncProgress(
                    namespace,
                    checkpoint.inventory_start_token or cursor,
                    pages_indexed,
                    items_indexed,
                    True,
                )

        return GDriveSyncProgress(
            namespace,
            checkpoint.inventory_start_token or cursor,
            pages_indexed,
            items_indexed,
            False,
        )

    async def resync_after_token_reset(self, scope: DriveScope) -> GDriveSyncProgress:
        """Replace an invalidated change cursor with one bounded inventory pass."""
        namespace = self._namespace(scope)
        start_token = (await self.source.client.get_start_page_token(scope)).token
        self.checkpoint_store.begin_inventory(namespace, start_token)
        progress = await self.sync_inventory(scope)
        return GDriveSyncProgress(
            progress.namespace,
            progress.start_token,
            progress.pages_indexed,
            progress.items_indexed,
            progress.complete,
            token_reset=True,
        )

    async def _materialize_changes(
        self,
        changes: Sequence[DriveChange],
        scope: DriveScope,
    ) -> list[Record]:
        records: list[Record] = []
        for change in changes:
            tombstone = self.source.tombstone_for_change(change, scope=scope)
            if tombstone is not None:
                records.append(tombstone)
            elif change.file is not None:
                records.append(
                    await self.source.materialize_record(change.file, scope=scope)
                )
        return records

    def _index_and_persist(self, records: Sequence[Record], kind: str) -> None:
        if not records:
            self.record_writer.persist()
            return
        for offset in range(0, len(records), self.batch_size):
            batch = records[offset : offset + self.batch_size]
            if self.record_writer.index_records(batch) is False:
                raise RuntimeError(f"Google Drive {kind} record indexing failed")
            self.record_writer.persist()

    def _namespace(self, scope: DriveScope) -> str:
        return checkpoint_namespace(
            f"{self.scope_generation}-{self.source.scope_identity(scope)}"
        )


def is_invalidated_change_token(error: BaseException) -> bool:
    """Recognize Drive responses that require a fresh change-feed start token."""
    info = classify_provider_error(error)
    if info.status_code == 410:
        return True
    reason = (info.reason or "").lower()
    message = info.message.lower()
    return reason in {"invalidpagetoken", "startpagenotfound", "pagetokenexpired"} or (
        "page token" in message
        and any(word in message for word in ("invalid", "expired", "not found"))
    )


__all__ = [
    "DriveRecordWriter",
    "GDriveSyncProgress",
    "GoogleDriveSync",
    "is_invalidated_change_token",
]
