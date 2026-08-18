"""Huey task registration for the Google Drive source lifecycle."""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from collections.abc import Coroutine, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Protocol, runtime_checkable
from uuid import uuid4

from huey import Huey
from searchkernel.api import ContentSource, Record

from mcp_markdown_ragdocs.coordination.task_leases import TaskLeasePort
from mcp_markdown_ragdocs.coordination.work_intents import WorkIntentPort
from mcp_markdown_ragdocs.gdrive.backfill import (
    GDriveBackfillCheckpointStore,
    GoogleDriveBackfill,
)
from mcp_markdown_ragdocs.gdrive.health import (
    DriveSourceHealth,
    GDriveHealthStore,
)
from mcp_markdown_ragdocs.gdrive.leases import DriveScopeLeaseStore
from mcp_markdown_ragdocs.gdrive.models import DriveScope
from mcp_markdown_ragdocs.gdrive.records import SOURCE_KIND
from mcp_markdown_ragdocs.gdrive.retry import DriveRetryWorkStore
from mcp_markdown_ragdocs.gdrive.retry import DRIVE_RETRY_OPERATION
from mcp_markdown_ragdocs.gdrive.checkpoints import GDriveSyncCheckpointStore
from mcp_markdown_ragdocs.gdrive.sync import GoogleDriveSync
from mcp_markdown_ragdocs.gdrive.watch import GDriveWatchStateStore, GoogleDriveWatch
from mcp_markdown_ragdocs.adapters.sources.gdrive import GoogleDriveContentSource
from mcp_markdown_ragdocs.config import Config

logger = logging.getLogger(__name__)

DRIVE_HEARTBEAT_INTERVAL_SECONDS = 5.0
GDRIVE_BACKGROUND_TASK_PRIORITY = -20
DRIVE_CONTINUE_DELAY_SECONDS = 1.0
DRIVE_RETRY_DELAY_SECONDS = 30.0
DRIVE_BACKFILL_DELAY_SECONDS = 3600.0
DRIVE_HEALTH_DELAY_SECONDS = 60.0


@runtime_checkable
class _FileHueyStorage(Protocol):
    filename: str


@runtime_checkable
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
    intents: WorkIntentPort
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


@dataclass
class GDriveLifecycleScheduler:
    """Deduplicate and schedule the durable Drive lifecycle transitions."""

    huey: Huey
    tasks: Mapping[str, object]
    runtime: GDriveTaskRuntime

    def startup(self, *, delay: float = 0.0) -> bool:
        return self._schedule("gdrive_startup", (), delay=delay)

    def inventory(self, scope_identity: str, *, delay: float = 0.0) -> bool:
        return self._schedule("gdrive_inventory", (scope_identity,), delay=delay)

    def changes(self, scope_identity: str, *, delay: float = 0.0) -> bool:
        return self._schedule("gdrive_changes", (scope_identity,), delay=delay)

    def retry(self, *, delay: float = DRIVE_RETRY_DELAY_SECONDS) -> bool:
        return self._schedule("gdrive_retry", (), delay=delay)

    def backfill(self, scope_identity: str, *, delay: float = DRIVE_BACKFILL_DELAY_SECONDS) -> bool:
        return self._schedule("gdrive_backfill", (scope_identity,), delay=delay)

    def watch(self, scope_identity: str, *, delay: float = 0.0) -> bool:
        return self._schedule("gdrive_watch", (scope_identity,), delay=delay)

    def health(self, *, delay: float = DRIVE_HEALTH_DELAY_SECONDS) -> bool:
        return self._schedule("gdrive_health", (), delay=delay)

    def _schedule(self, name: str, args: tuple[object, ...], *, delay: float) -> bool:
        task = self.tasks[name]
        task_name = name
        if _task_queued(self.huey, task_name, args):
            return False
        schedule = getattr(task, "schedule", None)
        if not callable(schedule):
            return False
        schedule(args=args, delay=delay)
        return True


def build_gdrive_task_runtime(
    manager: object,
    task_lease_store: TaskLeasePort,
    work_intent_store: WorkIntentPort,
) -> GDriveTaskRuntime | None:
    """Compose Drive lifecycle services when the logical source is enabled."""
    if not isinstance(manager, GDriveTaskManager):
        return None
    try:
        source = manager.get_content_source(SOURCE_KIND)
    except AttributeError:
        return None
    if not isinstance(source, GoogleDriveContentSource):
        return None

    config = manager._config
    drive_config = config.gdrive
    index_path = Path(manager.index_path)
    intents = work_intent_store
    retry = DriveRetryWorkStore(intents)
    source.retry_work_store = retry
    sync = GoogleDriveSync(
        source,
        GDriveSyncCheckpointStore(index_path),
        manager,
        scope_generation=drive_config.index_generation,
        page_size=drive_config.page_size,
        batch_size=drive_config.batch_size,
        max_items=drive_config.max_items,
        max_pages=drive_config.max_pages,
        max_seconds=drive_config.max_seconds,
        max_concurrent_materializations=drive_config.request_max_concurrent,
    )
    backfill = GoogleDriveBackfill(
        source,
        GDriveBackfillCheckpointStore(index_path),
        manager,
        scope_generation=drive_config.index_generation,
        page_size=drive_config.page_size,
        batch_size=drive_config.batch_size,
        max_items=drive_config.max_items,
        max_pages=drive_config.max_pages,
        max_seconds=drive_config.max_seconds,
    )
    leases = DriveScopeLeaseStore(task_lease_store)
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
        intents=intents,
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
    stop_heartbeat = threading.Event()
    ownership_lost = threading.Event()

    def heartbeat() -> None:
        while not stop_heartbeat.wait(DRIVE_HEARTBEAT_INTERVAL_SECONDS):
            if not runtime.leases.heartbeat(scope, owner_token):
                ownership_lost.set()
                return

    heartbeat_thread = threading.Thread(
        target=heartbeat,
        name=f"gdrive-lease-heartbeat-{scope_identity}",
        daemon=True,
    )
    heartbeat_thread.start()
    try:
        result = operation()
    except Exception as error:
        runtime.leases.fail(scope, owner_token, str(error))
        raise
    finally:
        stop_heartbeat.set()
        heartbeat_thread.join()
    if ownership_lost.is_set() or not runtime.leases.is_owner(scope, owner_token):
        runtime.leases.fail(scope, owner_token, "Drive scope lease ownership was lost")
        raise RuntimeError(f"Drive scope lease ownership was lost: {scope_identity}")
    if not runtime.leases.complete(scope, owner_token):
        raise RuntimeError(f"Drive scope lease completion failed: {scope_identity}")
    return {"status": "ok", "scope": scope_identity, "result": result}


def _task_queued(huey: Huey, task_name: str, args: tuple[object, ...]) -> bool:
    messages = []
    try:
        messages.extend(huey.storage.enqueued_items())
        messages.extend(huey.storage.scheduled_items())
    except Exception:
        logger.warning("Unable to inspect the Drive lifecycle queue", exc_info=True)
        return False
    for message in messages:
        try:
            task = huey.deserialize_task(message)
        except Exception:
            continue
        if getattr(task, "name", None) == task_name and tuple(getattr(task, "args", ())) == args:
            return True
    return False


def register_gdrive_tasks(
    huey: Huey,
    runtime: GDriveTaskRuntime | None,
) -> Mapping[str, object]:
    """Register and start the deduplicated Drive lifecycle on the Huey queue."""
    if runtime is None:
        return {}
    drive_runtime: GDriveTaskRuntime = runtime

    scheduler: GDriveLifecycleScheduler | None = None

    def inventory(scope_identity: str) -> dict[str, object]:
        try:
            result = _leased(
                runtime,
                scope_identity,
                lambda: asdict(_run(runtime.sync.sync_inventory(runtime.scope(scope_identity)))),
            )
        except Exception as error:
            runtime.health.record_sync_failure(
                runtime.source.workspace_id,
                scope_identity,
                str(error),
            )
            assert scheduler is not None
            scheduler.inventory(scope_identity, delay=DRIVE_RETRY_DELAY_SECONDS)
            scheduler.health()
            raise
        assert scheduler is not None
        progress = result.get("result")
        if isinstance(progress, dict) and progress.get("complete") is True:
            scheduler.changes(scope_identity)
            runtime.health.record_sync_success(
                runtime.source.workspace_id,
                scope_identity,
                indexed_records=runtime.manager.count_records(SOURCE_KIND),
            )
        else:
            scheduler.inventory(scope_identity, delay=DRIVE_CONTINUE_DELAY_SECONDS)
        scheduler.health()
        return result

    def changes(scope_identity: str) -> dict[str, object]:
        try:
            result = _leased(
                runtime,
                scope_identity,
                lambda: asdict(_run(runtime.sync.sync_changes(runtime.scope(scope_identity)))),
            )
        except Exception as error:
            runtime.health.record_sync_failure(
                runtime.source.workspace_id,
                scope_identity,
                str(error),
            )
            assert scheduler is not None
            scheduler.changes(scope_identity, delay=DRIVE_RETRY_DELAY_SECONDS)
            scheduler.health()
            raise
        assert scheduler is not None
        progress = result.get("result")
        if isinstance(progress, dict) and progress.get("complete") is True:
            scheduler.backfill(scope_identity)
            scheduler.watch(scope_identity)
        else:
            scheduler.changes(scope_identity, delay=DRIVE_CONTINUE_DELAY_SECONDS)
        runtime.health.record_sync_success(
            runtime.source.workspace_id,
            scope_identity,
            indexed_records=runtime.manager.count_records(SOURCE_KIND),
        )
        scheduler.health()
        return result

    def retry(intent_id: str | None = None) -> dict[str, object]:
        if intent_id is None:
            intent_id = next(
                (
                    intent.intent_id
                    for intent in runtime.intents.list_active()
                    if intent.operation == DRIVE_RETRY_OPERATION and intent.state == "pending"
                ),
                None,
            )
        if intent_id is None:
            assert scheduler is not None
            scheduler.retry()
            return {"status": "idle"}
        claimed = runtime.retry.claim(intent_id)
        if claimed is None:
            assert scheduler is not None
            scheduler.retry()
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
        assert scheduler is not None
        scheduler.retry()
        scheduler.health()
        return {"status": "ok", "intent_id": intent_id}

    def backfill(scope_identity: str) -> dict[str, object]:
        try:
            result = _leased(
                runtime,
                scope_identity,
                lambda: asdict(_run(runtime.backfill.run(runtime.scope(scope_identity), {}))),
            )
        except Exception as error:
            runtime.health.record_sync_failure(
                runtime.source.workspace_id,
                scope_identity,
                str(error),
            )
            assert scheduler is not None
            scheduler.backfill(scope_identity, delay=DRIVE_RETRY_DELAY_SECONDS)
            raise
        assert scheduler is not None
        progress = result.get("result")
        scheduler.backfill(
            scope_identity,
            delay=(DRIVE_BACKFILL_DELAY_SECONDS if isinstance(progress, dict) and progress.get("complete") else DRIVE_CONTINUE_DELAY_SECONDS),
        )
        scheduler.health()
        return result

    def lease(scope_identity: str, owner_token: str | None = None) -> dict[str, object]:
        scope = runtime.scope(scope_identity)
        token = owner_token or uuid4().hex
        claimed = runtime.leases.claim(scope, token)
        return {
            "status": "claimed" if claimed else "already_running",
            "scope": scope_identity,
            "owner_token": token,
        }

    def watch(scope_identity: str, page_token: str | None = None) -> dict[str, object]:
        scope = runtime.scope(scope_identity)
        if page_token is None:
            namespace = runtime.sync._namespace(scope)
            checkpoint = runtime.sync.checkpoint_store.load(namespace)
            page_token = checkpoint.changes_token if checkpoint is not None else None
        if page_token is None:
            assert scheduler is not None
            scheduler.watch(scope_identity, delay=DRIVE_RETRY_DELAY_SECONDS)
            return {"status": "not_ready", "scope": scope_identity}
        result = _leased(
            runtime,
            scope_identity,
            lambda: asdict(_run(runtime.watch.ensure(scope, page_token))),
        )
        assert scheduler is not None
        scheduler.watch(scope_identity, delay=runtime.watch.renewal_seconds)
        scheduler.health()
        return result

    def health() -> dict[str, object]:
        now = time.time()
        scopes = runtime.health.scopes_for(
            runtime.source.workspace_id,
            tuple(runtime.source.scope_identity(scope) for scope in runtime.source.scopes),
            indexed_records=runtime.manager.count_records(SOURCE_KIND),
        )
        previous = runtime.health.load(runtime.source.workspace_id)
        source = previous.get("source") if isinstance(previous, dict) else None
        available = source.get("available", True) if isinstance(source, dict) else True
        last_error = source.get("last_error") if isinstance(source, dict) else None
        evaluated = DriveSourceHealth.evaluate(
            runtime.source.workspace_id,
            scopes,
            available=bool(available),
            observed_at=now,
            watch_mode="push" if runtime.watch.push_enabled else "poll",
            last_error=last_error if isinstance(last_error, str) else None,
        )
        runtime.health.save(evaluated)
        assert scheduler is not None
        scheduler.health()
        return evaluated.to_payload()

    def startup() -> dict[str, object]:
        assert scheduler is not None
        scheduled = sum(
            scheduler.inventory(drive_runtime.source.scope_identity(scope))
            for scope in drive_runtime.source.scopes
        )
        scheduled += int(scheduler.retry())
        scheduled += int(scheduler.health(delay=0.0))
        return {"status": "ok", "scheduled": scheduled}

    handlers = {
        "gdrive_startup": startup,
        "gdrive_inventory": inventory,
        "gdrive_changes": changes,
        "gdrive_retry": retry,
        "gdrive_backfill": backfill,
        "gdrive_lease": lease,
        "gdrive_watch": watch,
        "gdrive_health": health,
    }
    registered = {
        name: huey.task(name=name, priority=GDRIVE_BACKGROUND_TASK_PRIORITY)(handler)
        for name, handler in handlers.items()
    }
    scheduler = GDriveLifecycleScheduler(huey, registered, runtime)
    scheduler.startup()
    return registered


__all__ = [
    "GDriveTaskRuntime",
    "GDriveLifecycleScheduler",
    "build_gdrive_task_runtime",
    "register_gdrive_tasks",
]
