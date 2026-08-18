"""Behavioral tests for the durable Google Drive state repository."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
import sqlite3

import pytest

from mcp_markdown_ragdocs.gdrive.adapter import (
    GDriveStateError,
    GDriveStateRepository,
    UnsupportedGDriveStateSchemaError,
)
from mcp_markdown_ragdocs.gdrive.domain import (
    GDriveBackfillCursor,
    GDriveCheckpoint,
    GDriveScopeIdentity,
    GDriveSyncStatus,
    GDriveWatchState,
)
from mcp_markdown_ragdocs.gdrive.port import GDriveStatePort


def _identity(scope: str = "shared-with-me") -> GDriveScopeIdentity:
    return GDriveScopeIdentity("google-drive", "workspace-a", scope)


def test_sqlite_adapter_satisfies_the_drive_state_port(tmp_path: Path) -> None:
    """
    Keep application consumers independent from the SQLite state adapter.
    """
    state: GDriveStatePort = GDriveStateRepository(tmp_path / "state.db")
    identity = _identity()

    assert isinstance(state, GDriveStatePort)
    assert state.add_membership(identity, "file-1") == ("shared-with-me",)
    assert state.memberships_for_source("google-drive", "workspace-a", "file-1") == (
        "shared-with-me",
    )


def test_state_survives_restart_for_each_record_type(tmp_path: Path) -> None:
    """
    Recover all durable state after reopening the same explicit database path.
    """
    path = tmp_path / "state" / "gdrive.db"
    identity = _identity()
    first = GDriveStateRepository(path)
    checkpoint = first.begin_inventory(identity, "start-token")
    first.persist_inventory_batch(identity, page_token="page-2", batch=1)
    first.persist_changes(identity, "changes-token")
    status = GDriveSyncStatus(identity, "healthy", last_success_at=12.5)
    backfill = GDriveBackfillCursor(identity, "generation-a", "page-1", 2)
    watch = GDriveWatchState(identity, "channel-1", "resource-1", 100, "https://example.test")
    first.save_sync_status(status)
    first.add_membership(identity, "file-1")
    first.save_backfill_cursor(backfill)
    first.save_watch_state(watch)

    second = GDriveStateRepository(path)

    assert second.load_checkpoint(identity) == replace(
        checkpoint,
        inventory_page_token="page-2",
        inventory_batch=1,
        changes_token="changes-token",
    )
    assert second.load_sync_status(identity) == status
    assert second.memberships_for(identity, "file-1") == ("shared-with-me",)
    assert second.load_backfill_cursor(identity, "generation-a") == backfill
    assert second.load_watch_state(identity) == watch


def test_memberships_are_duplicate_safe_and_removable(tmp_path: Path) -> None:
    """
    Keep memberships set-like while preserving workspace and source boundaries.
    """
    repository = GDriveStateRepository(tmp_path / "state.db")
    shared = _identity("shared-with-me")
    drive = _identity("shared-drive:drive-1")
    other_workspace = GDriveScopeIdentity("google-drive", "workspace-b", shared.scope_identity)
    other_source = GDriveScopeIdentity("other-source", "workspace-a", shared.scope_identity)

    assert repository.add_membership(shared, "file-1") == ("shared-with-me",)
    assert repository.add_membership(shared, "file-1") == ("shared-with-me",)
    assert repository.add_membership(drive, "file-1") == (
        "shared-drive:drive-1",
        "shared-with-me",
    )
    assert repository.memberships_for_source("google-drive", "workspace-a", "file-1") == (
        "shared-drive:drive-1",
        "shared-with-me",
    )
    assert repository.memberships_for(other_workspace, "file-1") == ()
    assert repository.memberships_for(other_source, "file-1") == ()
    assert repository.remove_membership(shared, "file-1") == ("shared-drive:drive-1",)
    assert repository.remove_membership(shared, "file-1") == ("shared-drive:drive-1",)
    assert repository.remove_membership(drive, "file-1") == ()


def test_read_modify_write_serializes_concurrent_checkpoint_updates(tmp_path: Path) -> None:
    """
    Serialize two repository instances without losing either checkpoint update.
    """
    path = tmp_path / "state.db"
    identity = _identity()
    first = GDriveStateRepository(path)
    first.begin_inventory(identity, "start-token")
    repositories = (first, GDriveStateRepository(path))

    def advance(repository: GDriveStateRepository) -> GDriveCheckpoint:
        def update(current: GDriveCheckpoint | None) -> GDriveCheckpoint:
            assert current is not None
            return replace(current, inventory_batch=current.inventory_batch + 1)

        return repository.update_checkpoint(identity, update)

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(advance, repositories))

    assert sorted(result.inventory_batch for result in results) == [1, 2]
    final = first.load_checkpoint(identity)
    assert final is not None
    assert final.inventory_batch == 2


def test_generation_and_scope_boundaries_are_enforced(tmp_path: Path) -> None:
    """
    Do not resume a backfill from another generation or another scope.
    """
    repository = GDriveStateRepository(tmp_path / "state.db")
    identity = _identity()
    other_scope = _identity("shared-drive:drive-1")
    repository.begin_backfill(identity, "generation-a")
    repository.persist_backfill_batch(
        identity,
        generation="generation-a",
        page_token="page-2",
        batch=1,
    )

    assert repository.load_backfill_cursor(identity, "generation-b") is None
    assert repository.load_backfill_cursor(other_scope, "generation-a") is None
    with pytest.raises(ValueError, match="generation must begin"):
        repository.persist_backfill_batch(
            other_scope,
            generation="generation-a",
            page_token=None,
            batch=1,
        )


def test_missing_and_malformed_state_are_safe(tmp_path: Path) -> None:
    """
    Treat missing rows and malformed record versions as deterministic empty state.
    """
    path = tmp_path / "state.db"
    identity = _identity()
    repository = GDriveStateRepository(path)
    assert repository.load_checkpoint(identity) is None

    with sqlite3.connect(path) as connection:
        connection.execute(
            "INSERT INTO checkpoints VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (*identity.as_parameters(), "start", None, 0, None, 99),
        )

    assert GDriveStateRepository(path).load_checkpoint(identity) is None


def test_unsupported_database_schema_is_rejected(tmp_path: Path) -> None:
    """
    Refuse an explicitly newer database schema instead of silently resetting it.
    """
    path = tmp_path / "state.db"
    repository = GDriveStateRepository(path)
    del repository
    with sqlite3.connect(path) as connection:
        connection.execute("PRAGMA user_version = 999")

    with pytest.raises(UnsupportedGDriveStateSchemaError, match="unsupported"):
        GDriveStateRepository(path)


def test_malformed_database_is_not_treated_as_valid_empty_state(tmp_path: Path) -> None:
    """
    Surface a corrupt SQLite file without deleting or overwriting user state.
    """
    path = tmp_path / "state.db"
    path.write_bytes(b"not a sqlite database")

    with pytest.raises(GDriveStateError, match="cannot open"):
        GDriveStateRepository(path)


def test_database_uses_wal_and_busy_timeout(tmp_path: Path) -> None:
    """
    Configure SQLite for concurrent readers and bounded writer waiting.
    """
    path = tmp_path / "state.db"
    repository = GDriveStateRepository(path, busy_timeout_ms=3210)

    with sqlite3.connect(path) as connection:
        assert connection.execute("PRAGMA journal_mode").fetchone() == ("wal",)
    assert repository.busy_timeout_ms == 3210
