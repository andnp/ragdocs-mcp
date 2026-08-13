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
from mcp_markdown_ragdocs.gdrive.models import DriveScope


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
        max_items: int = 100_000,
        max_pages: int = 500,
        max_seconds: float = 10.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not scope_generation or ":" in scope_generation:
            raise ValueError("scope_generation must be non-empty and must not contain ':'")
        if page_size < 1:
            raise ValueError("page_size must be positive")
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
        self.max_items = max_items
        self.max_pages = max_pages
        self.max_seconds = max_seconds
        self._clock = clock

    async def sync_inventory(self, scope: DriveScope) -> GDriveSyncProgress:
        """Index bounded inventory pages and advance after each durable write."""
        namespace = self._namespace(scope)
        checkpoint = self.checkpoint_store.load(namespace)
        if checkpoint is None or checkpoint.inventory_start_token is None:
            start_token = (await self.source.client.get_start_page_token(scope)).token
            checkpoint = self.checkpoint_store.begin_inventory(namespace, start_token)
        else:
            start_token = checkpoint.inventory_start_token

        if checkpoint.inventory_batch and checkpoint.inventory_page_token is None:
            return GDriveSyncProgress(namespace, start_token, 0, 0, True)

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
            if records and self.record_writer.index_records(records) is False:
                raise RuntimeError("Google Drive inventory record indexing failed")
            self.record_writer.persist()
            pages_indexed += 1
            items_indexed += len(records)
            checkpoint = self.checkpoint_store.persist_inventory_batch_after_index(
                namespace,
                page_token=page.next_page_token,
                batch=checkpoint.inventory_batch + 1,
            )
            page_token = page.next_page_token
            if page_token is None:
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

    def _namespace(self, scope: DriveScope) -> str:
        return checkpoint_namespace(
            f"{self.scope_generation}-{self.source.scope_identity(scope)}"
        )


__all__ = ["DriveRecordWriter", "GDriveSyncProgress", "GoogleDriveSync"]
