"""Tests for Google Drive lifecycle task registration."""

from pathlib import Path
from types import SimpleNamespace

from huey import SqliteHuey

from mcp_markdown_ragdocs.gdrive.tasks import (
    GDriveTaskRuntime,
    build_gdrive_task_runtime,
    register_gdrive_tasks,
)


def _runtime(tmp_path: Path) -> tuple[GDriveTaskRuntime, SqliteHuey]:
    queue = SqliteHuey(
        name="gdrive-tasks",
        filename=str(tmp_path / "queue.db"),
        immediate=False,
    )
    source = SimpleNamespace(
        source_kind="gdrive",
        scopes=(),
        workspace_id="workspace",
        retry_work_store=None,
        scope_identity=lambda scope: str(scope),
    )
    manager = SimpleNamespace(
        _config=SimpleNamespace(
            gdrive=SimpleNamespace(
                index_generation="gdrive-v1",
                page_size=10,
                max_items=20,
                max_pages=2,
                max_seconds=1.0,
                push_address="",
                push_enabled=False,
                watch_renewal_seconds=60,
            )
        ),
        index_path=tmp_path / "index",
        get_content_source=lambda _kind: source,
    )
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


def test_registers_the_seven_drive_lifecycle_tasks(tmp_path: Path) -> None:
    """
    Register inventory, change, retry, recovery, lease, watch, and health tasks.
    Use the existing Huey task registration boundary for worker discovery.
    """
    runtime, huey = _runtime(tmp_path)

    registered = register_gdrive_tasks(huey, runtime)

    assert set(registered) == {
        "gdrive_inventory",
        "gdrive_changes",
        "gdrive_retry",
        "gdrive_backfill",
        "gdrive_lease",
        "gdrive_watch",
        "gdrive_health",
    }
    assert all(task is not None for task in registered.values())
