"""Application-owned Google Drive synchronization state values."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math

STATE_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class GDriveScopeIdentity:
    """Identify one source, workspace, and synchronization scope."""

    source_kind: str
    workspace_id: str
    scope_identity: str

    def __post_init__(self) -> None:
        for name, value in (
            ("source_kind", self.source_kind),
            ("workspace_id", self.workspace_id),
            ("scope_identity", self.scope_identity),
        ):
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a non-empty string")

    def as_parameters(self) -> tuple[str, str, str]:
        """Return the identity values in database-key order."""

        return self.source_kind, self.workspace_id, self.scope_identity


@dataclass(frozen=True, slots=True)
class GDriveCheckpoint:
    """Versioned cursors for one Drive scope."""

    identity: GDriveScopeIdentity
    inventory_start_token: str | None = None
    inventory_page_token: str | None = None
    inventory_batch: int = 0
    changes_token: str | None = None
    schema_version: int = STATE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _validate_record_version(self.schema_version)
        _validate_optional_text(self.inventory_start_token, "inventory_start_token")
        _validate_optional_text(self.inventory_page_token, "inventory_page_token")
        _validate_optional_text(self.changes_token, "changes_token")
        _validate_non_negative_int(self.inventory_batch, "inventory_batch")

    def inventory_started(self, start_token: str) -> "GDriveCheckpoint":
        """Return a checkpoint for an inventory that is about to begin."""

        _validate_required_text(start_token, "start_token")
        return replace(
            self,
            inventory_start_token=start_token,
            inventory_page_token=None,
            inventory_batch=0,
            changes_token=None,
        )

    def inventory_batch_indexed(
        self,
        *,
        page_token: str | None,
        batch: int,
    ) -> "GDriveCheckpoint":
        """Return progress after one indexed inventory batch."""

        if self.inventory_start_token is None:
            raise ValueError("inventory must start before a batch is indexed")
        if batch != self.inventory_batch + 1:
            raise ValueError("inventory batches must advance in order")
        _validate_optional_text(page_token, "page_token")
        return replace(self, inventory_page_token=page_token, inventory_batch=batch)

    def changes_indexed(self, changes_token: str) -> "GDriveCheckpoint":
        """Return progress after one indexed changes batch."""

        if self.inventory_start_token is None:
            raise ValueError("inventory must start before changes are indexed")
        _validate_required_text(changes_token, "changes_token")
        return replace(self, changes_token=changes_token)


@dataclass(frozen=True, slots=True)
class GDriveSyncStatus:
    """Versioned evaluated status for one Drive scope."""

    identity: GDriveScopeIdentity
    status: str
    last_success_at: float | None = None
    last_error: str | None = None
    schema_version: int = STATE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _validate_record_version(self.schema_version)
        _validate_required_text(self.status, "status")
        if self.last_success_at is not None and not math.isfinite(self.last_success_at):
            raise ValueError("last_success_at must be finite or null")
        _validate_optional_text(self.last_error, "last_error")


@dataclass(frozen=True, slots=True)
class GDriveMembership:
    """Versioned visibility membership for one stable Drive record."""

    identity: GDriveScopeIdentity
    source_id: str
    schema_version: int = STATE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _validate_record_version(self.schema_version)
        _validate_required_text(self.source_id, "source_id")


@dataclass(frozen=True, slots=True)
class GDriveScopeMembershipSnapshot:
    """The durable source IDs visible in one Drive scope."""

    identity: GDriveScopeIdentity
    source_ids: tuple[str, ...] = ()
    schema_version: int = STATE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _validate_record_version(self.schema_version)
        if len(self.source_ids) != len(set(self.source_ids)):
            raise ValueError("scope membership source IDs must be unique")
        for source_id in self.source_ids:
            _validate_required_text(source_id, "source_id")


@dataclass(frozen=True, slots=True)
class GDriveBackfillCursor:
    """Versioned bounded backfill progress for one scope generation."""

    identity: GDriveScopeIdentity
    generation: str
    page_token: str | None = None
    batch: int = 0
    schema_version: int = STATE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _validate_record_version(self.schema_version)
        _validate_required_text(self.generation, "generation")
        _validate_optional_text(self.page_token, "page_token")
        _validate_non_negative_int(self.batch, "batch")


@dataclass(frozen=True, slots=True)
class GDriveWatchState:
    """Versioned push-watch state for one Drive scope."""

    identity: GDriveScopeIdentity
    channel_id: str
    resource_id: str | None
    expiration: int
    address: str
    status: str = "active"
    last_error: str | None = None
    schema_version: int = STATE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _validate_record_version(self.schema_version)
        _validate_required_text(self.channel_id, "channel_id")
        _validate_optional_text(self.resource_id, "resource_id")
        _validate_non_negative_int(self.expiration, "expiration")
        _validate_required_text(self.address, "address")
        _validate_required_text(self.status, "status")
        _validate_optional_text(self.last_error, "last_error")


def _validate_record_version(version: int) -> None:
    if version != STATE_SCHEMA_VERSION:
        raise ValueError(f"unsupported state schema version: {version}")


def _validate_required_text(value: str, name: str) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")


def _validate_optional_text(value: str | None, name: str) -> None:
    if value is not None:
        _validate_required_text(value, name)


def _validate_non_negative_int(value: int, name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")


__all__ = [
    "GDriveBackfillCursor",
    "GDriveCheckpoint",
    "GDriveMembership",
    "GDriveScopeIdentity",
    "GDriveScopeMembershipSnapshot",
    "GDriveSyncStatus",
    "GDriveWatchState",
    "STATE_SCHEMA_VERSION",
]
