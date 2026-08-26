"""Tests for Google Drive lifecycle task registration."""

from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import pytest
from huey import SqliteHuey
from huey.api import TaskWrapper
from searchkernel.api import ContentSource, Record

from mcp_markdown_ragdocs.adapters.sources.gdrive import (
    DriveContentClient,
    GoogleDriveContentSource,
)
from mcp_markdown_ragdocs.config import Config, GoogleDriveConfig
from mcp_markdown_ragdocs.gdrive.checkpoints import checkpoint_namespace
from mcp_markdown_ragdocs.gdrive.sync import GDriveSyncProgress
from mcp_markdown_ragdocs.gdrive.tasks import (
    DRIVE_CONTINUE_DELAY_SECONDS,
    DRIVE_INVENTORY_FAILURE_ATTEMPT_CAP,
    DRIVE_INVENTORY_TOKEN_POISON_THRESHOLD,
    DRIVE_RETRY_DELAY_CEILING_SECONDS,
    DRIVE_RETRY_DELAY_SECONDS,
    GDRIVE_BACKGROUND_TASK_PRIORITY,
    GDriveTaskRuntime,
    build_gdrive_task_runtime,
    register_gdrive_tasks,
)
from mcp_markdown_ragdocs.coordination.task_leases import TaskLeaseStore
from mcp_markdown_ragdocs.coordination.work_intents import WorkIntentStore
from mcp_markdown_ragdocs.gdrive.models import (
    DriveChangePage,
    DriveFile,
    DriveFilePage,
    DriveScope,
    DriveStartPageToken,
    DriveWatchChannel,
)


class _TaskClient(DriveContentClient):
    async def list_files_page(
        self,
        scope: DriveScope,
        *,
        page_token: str | None = None,
        page_size: int = 1000,
    ) -> DriveFilePage:
        raise AssertionError((scope, page_token, page_size))

    async def get_start_page_token(self, scope: DriveScope) -> DriveStartPageToken:
        raise AssertionError(scope)

    async def list_changes_page(
        self,
        scope: DriveScope,
        page_token: str,
        *,
        page_size: int = 1000,
    ) -> DriveChangePage:
        raise AssertionError((scope, page_token, page_size))

    async def export_file(self, file_id: str, export_mime_type: str) -> bytes:
        raise AssertionError((file_id, export_mime_type))

    async def download_file(self, file_id: str) -> bytes:
        raise AssertionError(file_id)

    async def get_file_metadata(self, file_id: str) -> DriveFile:
        raise AssertionError(file_id)

    async def watch_changes(
        self,
        scope: DriveScope,
        page_token: str,
        *,
        channel_id: str,
        address: str,
        token: str | None = None,
    ) -> DriveWatchChannel:
        raise AssertionError((scope, page_token, channel_id, address, token))

    async def stop_channel(self, channel_id: str, resource_id: str | None = None) -> None:
        raise AssertionError((channel_id, resource_id))


class _TaskManager:
    def __init__(self, index_path: Path, source: GoogleDriveContentSource) -> None:
        self._config = Config(
            gdrive=GoogleDriveConfig(
                index_generation="gdrive-v1",
                page_size=10,
                max_items=20,
                max_pages=2,
                max_seconds=1.0,
                push_address="",
                push_enabled=False,
                watch_renewal_seconds=60,
            )
        )
        self._index_path = index_path
        self._source = source

    @property
    def index_path(self) -> Path:
        return self._index_path

    def get_content_source(self, source_kind: str) -> ContentSource | None:
        return self._source if source_kind == "gdrive" else None

    def index_record(self, record: Record) -> bool:
        del record
        return True

    def index_records(self, records: Sequence[Record]) -> bool:
        del records
        return True

    def persist(self) -> None:
        return

    def count_records(self, source_kind: str | None = None) -> int:
        del source_kind
        return 0


def _runtime(tmp_path: Path) -> tuple[GDriveTaskRuntime, SqliteHuey]:
    queue = SqliteHuey(
        name="gdrive-tasks",
        filename=str(tmp_path / "queue.db"),
        immediate=False,
    )
    source = GoogleDriveContentSource(_TaskClient(), workspace_id="workspace")
    manager = _TaskManager(tmp_path / "index", source)
    runtime = build_gdrive_task_runtime(
        manager,
        TaskLeaseStore(tmp_path / "queue.db"),
        WorkIntentStore(tmp_path / "queue.db"),
    )
    assert runtime is not None
    return runtime, queue


def test_build_runtime_wires_existing_drive_components(tmp_path: Path) -> None:
    """
    Compose sync, retry, backfill, lease, watch, and health state together.
    Keep all durable state below the existing index or Huey queue roots.
    """
    runtime, _queue = _runtime(tmp_path)

    assert runtime.sync.source is runtime.source
    assert runtime.backfill.source is runtime.source
    assert runtime.source.retry_work_store is runtime.retry
    assert runtime.health.path == tmp_path / "index" / "gdrive-health.json"


def test_registers_the_drive_lifecycle_tasks(tmp_path: Path) -> None:
    """
    Register startup, inventory, change, retry, recovery, lease, watch, and health tasks.
    Use the existing Huey task registration boundary for worker discovery.
    """
    runtime, huey = _runtime(tmp_path)

    registered = register_gdrive_tasks(huey, runtime)

    assert set(registered) == {
        "gdrive_startup",
        "gdrive_inventory",
        "gdrive_changes",
        "gdrive_retry",
        "gdrive_backfill",
        "gdrive_lease",
        "gdrive_watch",
        "gdrive_health",
    }
    assert all(task is not None for task in registered.values())
    assert huey.pending_count() + huey.scheduled_count() == 1
    queued = huey.storage.enqueued_items() + huey.storage.scheduled_items()
    assert len(queued) == 1
    assert huey.deserialize_task(queued[0]).priority == GDRIVE_BACKGROUND_TASK_PRIORITY


def test_disabled_drive_registration_is_a_no_op(tmp_path: Path) -> None:
    """Leave the queue untouched when the Drive source is disabled."""
    huey = SqliteHuey(
        name="gdrive-disabled",
        filename=str(tmp_path / "queue.db"),
        immediate=False,
    )

    assert register_gdrive_tasks(huey, None) == {}
    assert huey.pending_count() == 0


class _FailingSync:
    """A sync boundary that always fails, to drive the retry/backoff path."""

    def __init__(self) -> None:
        self.calls = 0

    async def sync_inventory(self, scope: DriveScope) -> GDriveSyncProgress:
        del scope
        self.calls += 1
        raise RuntimeError("simulated inventory failure")


class _ProgressSync:
    """A sync boundary that returns one fixed inventory progress payload."""

    def __init__(self, progress: GDriveSyncProgress) -> None:
        self.progress = progress

    async def sync_inventory(self, scope: DriveScope) -> GDriveSyncProgress:
        del scope
        return self.progress


def _inventory_task(registered: Mapping[str, object]) -> TaskWrapper:
    return cast(TaskWrapper, registered["gdrive_inventory"])


def _inventory_namespace(runtime: GDriveTaskRuntime, scope_identity: str) -> str:
    return checkpoint_namespace(
        f"{runtime.sync.scope_generation}-{scope_identity}"
    )


def _scheduled_delay_seconds(huey: SqliteHuey, task_name: str, reference: datetime) -> float:
    """Return the eta delay of the single queued/scheduled task with this name."""

    messages = huey.storage.enqueued_items() + huey.storage.scheduled_items()
    matches = [
        task
        for task in (huey.deserialize_task(message) for message in messages)
        if task.name == task_name
    ]
    assert len(matches) == 1, f"expected exactly one {task_name} task, found {len(matches)}"
    return (matches[0].eta - reference).total_seconds()


def _has_task(huey: SqliteHuey, task_name: str) -> bool:
    messages = huey.storage.enqueued_items() + huey.storage.scheduled_items()
    return any(
        huey.deserialize_task(message).name == task_name for message in messages
    )


def test_inventory_failure_backoff_grows_with_consecutive_failures(tmp_path: Path) -> None:
    """
    Back off exponentially on repeated inventory failures instead of the
    fixed one-second self-reschedule that caused the runaway task loop.
    """
    runtime, huey = _runtime(tmp_path)
    registered = register_gdrive_tasks(huey, runtime)
    scope_identity = runtime.source.scope_identity(runtime.source.scopes[0])
    runtime.sync.sync_inventory = _FailingSync().sync_inventory  # type: ignore[method-assign]
    huey.storage.flush_queue()
    huey.storage.flush_schedule()

    expected_delays = (
        DRIVE_RETRY_DELAY_SECONDS,
        DRIVE_RETRY_DELAY_SECONDS * 2,
        DRIVE_RETRY_DELAY_SECONDS * 4,
    )
    for expected_delay in expected_delays:
        reference = datetime.now(UTC).replace(tzinfo=None)
        with pytest.raises(RuntimeError):
            _inventory_task(registered).call_local(scope_identity)
        delay = _scheduled_delay_seconds(huey, "gdrive_inventory", reference)
        assert delay == pytest.approx(expected_delay, abs=5.0)
        huey.storage.flush_queue()
        huey.storage.flush_schedule()


def test_inventory_failure_attempt_cap_stops_rescheduling(tmp_path: Path) -> None:
    """
    Stop rescheduling once a scope has failed too many times in a row,
    leaving it unhealthy instead of spinning forever.
    """
    runtime, huey = _runtime(tmp_path)
    registered = register_gdrive_tasks(huey, runtime)
    scope_identity = runtime.source.scope_identity(runtime.source.scopes[0])
    runtime.sync.sync_inventory = _FailingSync().sync_inventory  # type: ignore[method-assign]
    huey.storage.flush_queue()
    huey.storage.flush_schedule()

    for _ in range(DRIVE_INVENTORY_FAILURE_ATTEMPT_CAP - 1):
        with pytest.raises(RuntimeError):
            _inventory_task(registered).call_local(scope_identity)
        huey.storage.flush_queue()
        huey.storage.flush_schedule()

    with pytest.raises(RuntimeError):
        _inventory_task(registered).call_local(scope_identity)

    assert not _has_task(huey, "gdrive_inventory")

    namespace = _inventory_namespace(runtime, scope_identity)
    checkpoint = runtime.sync.checkpoint_store.load(namespace)
    assert checkpoint is not None
    assert checkpoint.inventory_failure_count == DRIVE_INVENTORY_FAILURE_ATTEMPT_CAP


def test_inventory_backoff_is_capped_at_the_ceiling(tmp_path: Path) -> None:
    """Never let the backoff delay itself grow past the configured ceiling."""
    runtime, huey = _runtime(tmp_path)
    registered = register_gdrive_tasks(huey, runtime)
    scope_identity = runtime.source.scope_identity(runtime.source.scopes[0])
    runtime.sync.sync_inventory = _FailingSync().sync_inventory  # type: ignore[method-assign]
    huey.storage.flush_queue()
    huey.storage.flush_schedule()

    reference = datetime.now(UTC).replace(tzinfo=None)
    for step in range(DRIVE_INVENTORY_FAILURE_ATTEMPT_CAP - 1):
        reference = datetime.now(UTC).replace(tzinfo=None)
        with pytest.raises(RuntimeError):
            _inventory_task(registered).call_local(scope_identity)
        if step < DRIVE_INVENTORY_FAILURE_ATTEMPT_CAP - 2:
            huey.storage.flush_queue()
            huey.storage.flush_schedule()

    delay = _scheduled_delay_seconds(huey, "gdrive_inventory", reference)
    assert delay <= DRIVE_RETRY_DELAY_CEILING_SECONDS + 5.0


def test_poisoned_inventory_token_is_cleared_after_repeated_identical_failures(
    tmp_path: Path,
) -> None:
    """
    Stop replaying a page token that keeps failing (e.g. a Drive 500) by
    clearing it so pagination restarts from the beginning of the scope.
    """
    runtime, huey = _runtime(tmp_path)
    registered = register_gdrive_tasks(huey, runtime)
    scope_identity = runtime.source.scope_identity(runtime.source.scopes[0])
    namespace = _inventory_namespace(runtime, scope_identity)
    runtime.sync.checkpoint_store.begin_inventory(namespace, "start-token")
    runtime.sync.checkpoint_store.persist_inventory_batch_after_index(
        namespace, page_token="poisoned-token", batch=1
    )
    runtime.sync.sync_inventory = _FailingSync().sync_inventory  # type: ignore[method-assign]
    huey.storage.flush_queue()
    huey.storage.flush_schedule()

    for _ in range(DRIVE_INVENTORY_TOKEN_POISON_THRESHOLD - 1):
        with pytest.raises(RuntimeError):
            _inventory_task(registered).call_local(scope_identity)
        huey.storage.flush_queue()
        huey.storage.flush_schedule()

    checkpoint_before = runtime.sync.checkpoint_store.load(namespace)
    assert checkpoint_before is not None
    assert checkpoint_before.inventory_page_token == "poisoned-token"

    with pytest.raises(RuntimeError):
        _inventory_task(registered).call_local(scope_identity)

    checkpoint_after = runtime.sync.checkpoint_store.load(namespace)
    assert checkpoint_after is not None
    assert checkpoint_after.inventory_page_token is None
    assert checkpoint_after.inventory_start_token == "start-token"


def test_inventory_progress_reschedules_promptly_and_clears_failure_history(
    tmp_path: Path,
) -> None:
    """
    Keep the fast one-second continuation delay when a run actually makes
    progress, and forget any prior failure streak once it succeeds.
    """
    runtime, huey = _runtime(tmp_path)
    registered = register_gdrive_tasks(huey, runtime)
    scope_identity = runtime.source.scope_identity(runtime.source.scopes[0])
    namespace = _inventory_namespace(runtime, scope_identity)
    runtime.sync.sync_inventory = _FailingSync().sync_inventory  # type: ignore[method-assign]
    huey.storage.flush_queue()
    huey.storage.flush_schedule()
    with pytest.raises(RuntimeError):
        _inventory_task(registered).call_local(scope_identity)
    huey.storage.flush_queue()
    huey.storage.flush_schedule()
    checkpoint_after_failure = runtime.sync.checkpoint_store.load(namespace)
    assert checkpoint_after_failure is not None
    assert checkpoint_after_failure.inventory_failure_count == 1

    progress = GDriveSyncProgress(
        namespace=namespace,
        start_token="start-token",
        pages_indexed=1,
        items_indexed=5,
        complete=False,
    )
    runtime.sync.sync_inventory = _ProgressSync(progress).sync_inventory  # type: ignore[method-assign]
    reference = datetime.now(UTC).replace(tzinfo=None)

    _inventory_task(registered).call_local(scope_identity)

    delay = _scheduled_delay_seconds(huey, "gdrive_inventory", reference)
    assert delay == pytest.approx(DRIVE_CONTINUE_DELAY_SECONDS, abs=2.0)
    checkpoint_after_success = runtime.sync.checkpoint_store.load(namespace)
    assert checkpoint_after_success is not None
    assert checkpoint_after_success.inventory_failure_count == 0


def test_inventory_progress_with_no_pages_backs_off_instead_of_spinning(
    tmp_path: Path,
) -> None:
    """A run that indexed nothing must not re-fire at the one-second delay."""
    runtime, huey = _runtime(tmp_path)
    registered = register_gdrive_tasks(huey, runtime)
    scope_identity = runtime.source.scope_identity(runtime.source.scopes[0])
    namespace = _inventory_namespace(runtime, scope_identity)
    progress = GDriveSyncProgress(
        namespace=namespace,
        start_token="start-token",
        pages_indexed=0,
        items_indexed=0,
        complete=False,
    )
    runtime.sync.sync_inventory = _ProgressSync(progress).sync_inventory  # type: ignore[method-assign]
    huey.storage.flush_queue()
    huey.storage.flush_schedule()
    reference = datetime.now(UTC).replace(tzinfo=None)

    _inventory_task(registered).call_local(scope_identity)

    delay = _scheduled_delay_seconds(huey, "gdrive_inventory", reference)
    assert delay == pytest.approx(DRIVE_RETRY_DELAY_SECONDS, abs=2.0)


def test_completing_inventory_schedules_changes_instead_of_more_inventory(
    tmp_path: Path,
) -> None:
    """Preserve the existing transition from a finished inventory to changes."""
    runtime, huey = _runtime(tmp_path)
    registered = register_gdrive_tasks(huey, runtime)
    scope_identity = runtime.source.scope_identity(runtime.source.scopes[0])
    namespace = _inventory_namespace(runtime, scope_identity)
    progress = GDriveSyncProgress(
        namespace=namespace,
        start_token="start-token",
        pages_indexed=1,
        items_indexed=1,
        complete=True,
    )
    runtime.sync.sync_inventory = _ProgressSync(progress).sync_inventory  # type: ignore[method-assign]
    huey.storage.flush_queue()
    huey.storage.flush_schedule()

    _inventory_task(registered).call_local(scope_identity)

    assert _has_task(huey, "gdrive_changes")
    assert not _has_task(huey, "gdrive_inventory")
