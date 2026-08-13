"""Huey task registration for the Google Drive source lifecycle."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Coroutine, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Protocol, runtime_checkable
from uuid import uuid4

from huey import Huey
from searchkernel.api import ContentSource, Record

from mcp_markdown_ragdocs.coordination.task_leases import TaskLeaseStore
from mcp_markdown_ragdocs.coordination.work_intents import WorkIntentStore
from mcp_markdown_ragdocs.gdrive.backfill import (
    GDriveBackfillCheckpointStore,
    GoogleDriveBackfill,
)
from mcp_markdown_ragdocs.gdrive.health import (
    DriveScopeHealth,
    DriveSourceHealth,
    GDriveHealthStore,
)
from mcp_markdown_ragdocs.gdrive.leases import DriveScopeLeaseStore
from mcp_markdown_ragdocs.gdrive.models import DriveScope
from mcp_markdown_ragdocs.gdrive.records import SOURCE_KIND
from mcp_markdown_ragdocs.gdrive.retry import DriveRetryWorkStore
from mcp_markdown_ragdocs.gdrive.checkpoints import GDriveSyncCheckpointStore
from mcp_markdown_ragdocs.gdrive.sync import GoogleDriveSync
from mcp_markdown_ragdocs.gdrive.watch import GDriveWatchStateStore, GoogleDriveWatch
from mcp_markdown_ragdocs.adapters.sources.gdrive import GoogleDriveContentSource
from mcp_markdown_ragdocs.config import Config
from mcp_markdown_ragdocs.indexing.task_registration import register_huey_tasks


@runtime_checkable
class _FileHueyStorage(Protocol):
    filename: str


class GDriveTaskManager(Protocol):
    _config: Config

    @property
    def index_path(self) -> Path: ...

    def get_content_source(self, source_kind: str) -> ContentSource | None: ...

    def index_record(self, record: Record) -> bool: ...

    def index_records(self, records: Sequence[Record]) -> bool: ...

    def persist(self) -> None: ...

    def count_records(self, source_kind: str | None = None) -> int: ...


@dataclass
class GDriveTaskRuntime:
    """Drive operations and their source-specific durable state."""

    manager: GDriveTaskManager
    source: GoogleDriveContentSource
    sync: GoogleDriveSync
    retry: DriveRetryWorkStore
    backfill: GoogleDriveBackfill
    leases: DriveScopeLeaseStore
    watch: GoogleDriveWatch
    health: GDriveHealthStore

    def scope(self, scope_identity: str) -> DriveScope:
        for scope in self.source.scopes:
            if self.source.scope_identity(scope) == scope_identity:
                return scope
        raise KeyError(f"unknown Drive scope: {scope_identity!r}")


def build_gdrive_task_runtime(
    manager: GDriveTaskManager,
    huey: Huey,
) -> GDriveTaskRuntime | None:
    """Compose Drive lifecycle services when the logical source is enabled."""
    source = manager.get_content_source(SOURCE_KIND)
    if not isinstance(source, GoogleDriveContentSource):
        return None

    config = manager._config
    drive_config = config.gdrive
    index_path = Path(manager.index_path)
    storage = huey.storage
    if not isinstance(storage, _FileHueyStorage):
        return None
    queue_path = Path(storage.filename)
    retry = DriveRetryWorkStore(WorkIntentStore(queue_path))
    source.retry_work_store = retry
    sync = GoogleDriveSync(
        source,
        GDriveSyncCheckpointStore(index_path),
        manager,
        scope_generation=drive_config.index_generation,
        page_size=drive_config.page_size,
        max_items=drive_config.max_items,
        max_pages=drive_config.max_pages,
        max_seconds=drive_config.max_seconds,
    )
    backfill = GoogleDriveBackfill(
        source,
        GDriveBackfillCheckpointStore(index_path),
        manager,
        scope_generation=drive_config.index_generation,
        page_size=drive_config.page_size,
        max_items=drive_config.max_items,
        max_pages=drive_config.max_pages,
        max_seconds=drive_config.max_seconds,
    )
    leases = DriveScopeLeaseStore(TaskLeaseStore(queue_path))
    watch = GoogleDriveWatch(
        source,
        GDriveWatchStateStore(index_path),
        scope_generation=drive_config.index_generation,
        address=drive_config.push_address,
        renewal_seconds=drive_config.watch_renewal_seconds,
        push_enabled=drive_config.push_enabled,
        retry_work_store=retry,
    )
    return GDriveTaskRuntime(
        manager=manager,
        source=source,
        sync=sync,
        retry=retry,
        backfill=backfill,
        leases=leases,
        watch=watch,
        health=GDriveHealthStore(index_path),
    )


def _run[T](coroutine: Coroutine[object, object, T]) -> T:
    return asyncio.run(coroutine)


def _leased(
    runtime: GDriveTaskRuntime,
    scope_identity: str,
    operation: Callable[[], object],
) -> dict[str, object]:
    scope = runtime.scope(scope_identity)
    owner_token = uuid4().hex
    if not runtime.leases.claim(scope, owner_token):
        return {"status": "already_running", "scope": scope_identity}
    try:
        result = operation()
    except Exception as error:
        runtime.leases.fail(scope, owner_token, str(error))
        raise
    runtime.leases.complete(scope, owner_token)
    return {"status": "ok", "scope": scope_identity, "result": result}


def register_gdrive_tasks(
    huey: Huey,
    runtime: GDriveTaskRuntime | None,
) -> dict[str, object]:
    """Register the seven Drive lifecycle tasks on the existing Huey queue."""
    if runtime is None:
        return {}

    def inventory(scope_identity: str) -> dict[str, object]:
        return _leased(
            runtime,
            scope_identity,
            lambda: asdict(_run(runtime.sync.sync_inventory(runtime.scope(scope_identity)))),
        )

    def changes(scope_identity: str) -> dict[str, object]:
        return _leased(
            runtime,
            scope_identity,
            lambda: asdict(_run(runtime.sync.sync_changes(runtime.scope(scope_identity)))),
        )

    def retry(intent_id: str) -> dict[str, object]:
        claimed = runtime.retry.claim(intent_id)
        if claimed is None:
            return {"status": "not_claimed", "intent_id": intent_id}
        work, claim_token = claimed
        scope = runtime.scope(work.scope_identity)
        try:
            file = _run(runtime.source.client.get_file_metadata(work.source_id))
            record = _run(runtime.source.materialize_record(file, scope=scope))
            if not runtime.manager.index_record(record):
                raise RuntimeError("Google Drive retry indexing failed")
            runtime.manager.persist()
        except Exception as error:
            runtime.retry.retry(intent_id, claim_token, error)
            raise
        runtime.retry.complete(intent_id, claim_token)
        return {"status": "ok", "intent_id": intent_id}

    def backfill(scope_identity: str) -> dict[str, object]:
        return _leased(
            runtime,
            scope_identity,
            lambda: asdict(_run(runtime.backfill.run(runtime.scope(scope_identity), {}))),
        )

    def lease(scope_identity: str, owner_token: str | None = None) -> dict[str, object]:
        scope = runtime.scope(scope_identity)
        token = owner_token or uuid4().hex
        claimed = runtime.leases.claim(scope, token)
        return {
            "status": "claimed" if claimed else "already_running",
            "scope": scope_identity,
            "owner_token": token,
        }

    def watch(scope_identity: str, page_token: str) -> dict[str, object]:
        return _leased(
            runtime,
            scope_identity,
            lambda: asdict(_run(runtime.watch.ensure(runtime.scope(scope_identity), page_token))),
        )

    def health() -> dict[str, object]:
        now = time.time()
        scopes = tuple(
            DriveScopeHealth(
                runtime.source.scope_identity(scope),
                indexed_records=runtime.manager.count_records(SOURCE_KIND),
                last_success_at=now,
            )
            for scope in runtime.source.scopes
        )
        evaluated = DriveSourceHealth.evaluate(
            runtime.source.workspace_id,
            scopes,
            observed_at=now,
            watch_mode="push" if runtime.watch.push_enabled else "poll",
        )
        runtime.health.save(evaluated)
        return evaluated.to_payload()

    return register_huey_tasks(
        huey,
        {
            "gdrive_inventory": inventory,
            "gdrive_changes": changes,
            "gdrive_retry": retry,
            "gdrive_backfill": backfill,
            "gdrive_lease": lease,
            "gdrive_watch": watch,
            "gdrive_health": health,
        },
    )


__all__ = [
    "GDriveTaskRuntime",
    "build_gdrive_task_runtime",
    "register_gdrive_tasks",
]
