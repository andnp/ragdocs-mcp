"""Per-scope execution leases for Google Drive synchronization."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass

from mcp_markdown_ragdocs.coordination.task_leases import (
    ACTIVE_LEASE,
    DEFAULT_LEASE_TIMEOUT_SECONDS,
    TaskLease,
    TaskLeaseStore,
)
from mcp_markdown_ragdocs.gdrive.models import DriveScope

DRIVE_SCOPE_LEASE_NAME = "gdrive_scope_sync"


@dataclass(frozen=True, slots=True)
class DriveScopeLease:
    """A generic task lease identified by a Drive workspace and scope."""

    task_id: str
    workspace_id: str
    scope_identity: str
    lease: TaskLease


class DriveScopeLeaseStore:
    """Adapt task leases to the one-owner-per-Drive-scope contract."""

    def __init__(self, leases: TaskLeaseStore) -> None:
        self._leases = leases
        self._timeout_seconds = float(
            getattr(leases, "_timeout_seconds", DEFAULT_LEASE_TIMEOUT_SECONDS)
        )

    def claim(
        self,
        scope: DriveScope,
        owner_token: str,
        *,
        now: float | None = None,
    ) -> bool:
        task_id = scope_task_id(scope)
        return self._leases.claim(
            task_id,
            task_name=DRIVE_SCOPE_LEASE_NAME,
            owner_token=owner_token,
            payload=json.dumps(scope_payload(scope), sort_keys=True).encode(),
            now=now,
        )

    def heartbeat(
        self,
        scope: DriveScope,
        owner_token: str,
        *,
        now: float | None = None,
    ) -> bool:
        if not self.is_owner(scope, owner_token, now=now):
            return False
        return self._leases.heartbeat(scope_task_id(scope), owner_token=owner_token, now=now)

    def is_owner(
        self,
        scope: DriveScope,
        owner_token: str,
        *,
        now: float | None = None,
    ) -> bool:
        """Return whether the owner still holds a live lease for the scope."""
        timestamp = time.time() if now is None else now
        lease = self.get(scope)
        return bool(
            lease is not None
            and lease.lease.state == ACTIVE_LEASE
            and lease.lease.owner_token == owner_token
            and lease.lease.heartbeat_at > timestamp - self._timeout_seconds
        )

    def complete(
        self,
        scope: DriveScope,
        owner_token: str,
        *,
        now: float | None = None,
    ) -> bool:
        return self._leases.complete(scope_task_id(scope), owner_token=owner_token, now=now)

    def fail(
        self,
        scope: DriveScope,
        owner_token: str,
        error: str,
        *,
        now: float | None = None,
    ) -> bool:
        return self._leases.fail(
            scope_task_id(scope), owner_token=owner_token, error=error, now=now
        )

    def get(self, scope: DriveScope) -> DriveScopeLease | None:
        task_id = scope_task_id(scope)
        lease = self._leases.get(task_id)
        if lease is None or lease.task_name != DRIVE_SCOPE_LEASE_NAME:
            return None
        return DriveScopeLease(
            task_id=task_id,
            workspace_id=scope.workspace_id,
            scope_identity=scope_identity(scope),
            lease=lease,
        )


def scope_identity(scope: DriveScope) -> str:
    if scope.shared_drive_id:
        return f"shared-drive:{scope.shared_drive_id}"
    if scope.include_shared_with_me:
        return "shared-with-me"
    raise ValueError("Drive scope must identify shared-with-me or a shared drive")


def scope_task_id(scope: DriveScope) -> str:
    return f"gdrive-scope:{scope.workspace_id}:{scope_identity(scope)}"


def scope_payload(scope: DriveScope) -> dict[str, str]:
    return {
        "workspace_id": scope.workspace_id,
        "scope_identity": scope_identity(scope),
    }


__all__ = [
    "DRIVE_SCOPE_LEASE_NAME",
    "DriveScopeLease",
    "DriveScopeLeaseStore",
    "scope_identity",
    "scope_payload",
    "scope_task_id",
]
