"""Tests for Google Drive scope lease recovery."""

from pathlib import Path

from mcp_markdown_ragdocs.coordination.task_leases import TaskLeaseStore
from mcp_markdown_ragdocs.gdrive.leases import DriveScopeLeaseStore, scope_task_id
from mcp_markdown_ragdocs.gdrive.models import DriveScope


def _scopes() -> tuple[DriveScope, DriveScope]:
    return (
        DriveScope("workspace", include_shared_with_me=True),
        DriveScope("workspace", shared_drive_id="drive-1"),
    )


def _store(tmp_path: Path) -> DriveScopeLeaseStore:
    return DriveScopeLeaseStore(TaskLeaseStore(tmp_path / "queue.db", timeout_seconds=10))


def test_scope_leases_use_distinct_stable_tasks(tmp_path: Path) -> None:
    """
    Keep shared-with-me and shared-drive synchronization ownership separate.
    """
    shared_with_me, shared_drive = _scopes()
    store = _store(tmp_path)

    assert scope_task_id(shared_with_me) != scope_task_id(shared_drive)
    assert store.claim(shared_with_me, "owner-a", now=10)
    assert store.claim(shared_drive, "owner-b", now=10)
    assert store.claim(shared_with_me, "owner-b", now=11) is False


def test_expired_scope_lease_allows_deterministic_takeover(tmp_path: Path) -> None:
    """
    Let a new worker reclaim a scope after the prior owner stops heartbeating.
    """
    scope = _scopes()[0]
    store = _store(tmp_path)

    assert store.claim(scope, "owner-a", now=10)
    assert store.claim(scope, "owner-b", now=19) is False
    assert store.claim(scope, "owner-b", now=20) is True
    lease = store.get(scope)

    assert lease is not None
    assert lease.lease.owner_token == "owner-b"
    assert lease.lease.attempt == 2


def test_stale_owner_cannot_finish_recovered_scope(tmp_path: Path) -> None:
    """
    Prevent an expired owner from completing work after another owner takes over.
    """
    scope = _scopes()[0]
    store = _store(tmp_path)

    assert store.claim(scope, "owner-a", now=10)
    assert store.claim(scope, "owner-b", now=20)

    assert store.heartbeat(scope, "owner-a", now=21) is False
    assert store.complete(scope, "owner-a", now=21) is False
    assert store.complete(scope, "owner-b", now=21) is True


def test_expired_owner_cannot_revive_a_scope_by_heartbeating(tmp_path: Path) -> None:
    """Reject a heartbeat after the lease timeout even before takeover."""
    scope = _scopes()[0]
    store = _store(tmp_path)

    assert store.claim(scope, "owner-a", now=10)

    assert store.heartbeat(scope, "owner-a", now=20) is False
    assert store.is_owner(scope, "owner-a", now=20) is False
