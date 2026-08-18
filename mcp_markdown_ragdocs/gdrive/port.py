"""Application-facing ports for durable Google Drive synchronization state."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Protocol, runtime_checkable

from mcp_markdown_ragdocs.gdrive.domain import (
    GDriveCheckpoint,
    GDriveScopeIdentity,
    GDriveScopeMembershipSnapshot,
    GDriveSyncStatus,
)

CheckpointUpdater = Callable[[GDriveCheckpoint | None], GDriveCheckpoint]


@runtime_checkable
class GDriveStatePort(Protocol):
    """Application capabilities required from durable Drive state."""

    def load_checkpoint(self, identity: GDriveScopeIdentity) -> GDriveCheckpoint | None: ...

    def save_checkpoint(self, checkpoint: GDriveCheckpoint) -> None: ...

    def update_checkpoint(
        self,
        identity: GDriveScopeIdentity,
        updater: CheckpointUpdater,
    ) -> GDriveCheckpoint: ...

    def begin_inventory(
        self,
        identity: GDriveScopeIdentity,
        start_token: str,
    ) -> GDriveCheckpoint: ...

    def persist_inventory_batch(
        self,
        identity: GDriveScopeIdentity,
        *,
        page_token: str | None,
        batch: int,
    ) -> GDriveCheckpoint: ...

    def persist_changes(
        self,
        identity: GDriveScopeIdentity,
        changes_token: str,
    ) -> GDriveCheckpoint: ...

    def load_sync_status(self, identity: GDriveScopeIdentity) -> GDriveSyncStatus | None: ...

    def save_sync_status(self, status: GDriveSyncStatus) -> None: ...

    def add_membership(
        self,
        identity: GDriveScopeIdentity,
        source_id: str,
    ) -> tuple[str, ...]: ...

    def load_scope_memberships(
        self,
        identity: GDriveScopeIdentity,
    ) -> GDriveScopeMembershipSnapshot: ...

    def replace_scope_memberships(
        self,
        identity: GDriveScopeIdentity,
        source_ids: Iterable[str],
    ) -> tuple[str, ...]: ...

    def remove_membership(
        self,
        identity: GDriveScopeIdentity,
        source_id: str,
    ) -> tuple[str, ...]: ...

    def memberships_for_source(
        self,
        source_kind: str,
        workspace_id: str,
        source_id: str,
    ) -> tuple[str, ...]: ...


__all__ = ["CheckpointUpdater", "GDriveStatePort"]
