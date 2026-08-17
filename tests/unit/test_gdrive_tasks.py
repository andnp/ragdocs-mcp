"""Tests for Google Drive lifecycle task registration."""

from pathlib import Path
from collections.abc import Sequence

from huey import SqliteHuey
from searchkernel.api import ContentSource, Record

from mcp_markdown_ragdocs.adapters.sources.gdrive import (
    DriveContentClient,
    GoogleDriveContentSource,
)
from mcp_markdown_ragdocs.config import Config, GoogleDriveConfig
from mcp_markdown_ragdocs.gdrive.tasks import (
    GDRIVE_BACKGROUND_TASK_PRIORITY,
    GDriveTaskRuntime,
    build_gdrive_task_runtime,
    register_gdrive_tasks,
)
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
    runtime = build_gdrive_task_runtime(manager, queue)
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
