"""Compatibility imports for the Google Drive state boundary.

New code should import domain values, the application port, or the SQLite
adapter from their dedicated modules.
"""

from mcp_markdown_ragdocs.gdrive.adapter import (
    DEFAULT_BUSY_TIMEOUT_MS,
    STATE_SCHEMA_VERSION,
    GDriveStateError,
    GDriveStateRepository,
    UnsupportedGDriveStateSchemaError,
)
from mcp_markdown_ragdocs.gdrive.domain import (
    GDriveBackfillCursor,
    GDriveCheckpoint,
    GDriveMembership,
    GDriveScopeIdentity,
    GDriveScopeMembershipSnapshot,
    GDriveSyncStatus,
    GDriveWatchState,
)
from mcp_markdown_ragdocs.gdrive.port import GDriveStatePort

__all__ = [
    "DEFAULT_BUSY_TIMEOUT_MS",
    "STATE_SCHEMA_VERSION",
    "GDriveBackfillCursor",
    "GDriveCheckpoint",
    "GDriveMembership",
    "GDriveScopeIdentity",
    "GDriveScopeMembershipSnapshot",
    "GDriveStateError",
    "GDriveStatePort",
    "GDriveStateRepository",
    "GDriveSyncStatus",
    "GDriveWatchState",
    "UnsupportedGDriveStateSchemaError",
]
