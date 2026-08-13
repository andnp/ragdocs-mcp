"""Versioned Google Drive synchronization checkpoint values."""

from __future__ import annotations

from dataclasses import dataclass, replace
CHECKPOINT_SCHEMA_VERSION = 1
GDRIVE_CHECKPOINT_NAMESPACE_PREFIX = "gdrive-v1"


def checkpoint_namespace(scope_generation: str) -> str:
    """Return the source-specific namespace for one Drive scope generation."""

    if not scope_generation or ":" in scope_generation:
        raise ValueError("scope_generation must be non-empty and must not contain ':'")
    return f"{GDRIVE_CHECKPOINT_NAMESPACE_PREFIX}:{scope_generation}"


@dataclass(frozen=True, slots=True)
class GDriveSyncCheckpoint:
    """Durable cursors for the bounded phases of a Drive synchronization."""

    inventory_start_token: str | None = None
    inventory_page_token: str | None = None
    inventory_batch: int = 0
    changes_token: str | None = None
    schema_version: int = CHECKPOINT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != CHECKPOINT_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported Google Drive checkpoint schema: {self.schema_version}"
            )
        if not isinstance(self.inventory_batch, int) or isinstance(self.inventory_batch, bool):
            raise ValueError("inventory_batch must be a non-negative integer")
        if self.inventory_batch < 0:
            raise ValueError("inventory_batch must be a non-negative integer")
        for field_name in (
            "inventory_start_token",
            "inventory_page_token",
            "changes_token",
        ):
            value = getattr(self, field_name)
            if value is not None and (not isinstance(value, str) or not value):
                raise ValueError(f"{field_name} must be a non-empty string or null")

    def to_payload(self) -> dict[str, object]:
        """Return the complete versioned JSON representation."""

        return {
            "schema_version": self.schema_version,
            "inventory_start_token": self.inventory_start_token,
            "inventory_page_token": self.inventory_page_token,
            "inventory_batch": self.inventory_batch,
            "changes_token": self.changes_token,
        }

    @classmethod
    def from_payload(cls, payload: object) -> "GDriveSyncCheckpoint":
        """Build a checkpoint from a versioned JSON object."""

        if not isinstance(payload, dict):
            raise ValueError("Google Drive checkpoint must be a JSON object")
        return cls(
            schema_version=payload.get("schema_version"),
            inventory_start_token=payload.get("inventory_start_token"),
            inventory_page_token=payload.get("inventory_page_token"),
            inventory_batch=payload.get("inventory_batch"),
            changes_token=payload.get("changes_token"),
        )

    def inventory_started(self, start_token: str) -> "GDriveSyncCheckpoint":
        """Return the checkpoint that must be saved before inventory begins."""

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
    ) -> "GDriveSyncCheckpoint":
        """Return progress after an inventory batch has been indexed."""

        if self.inventory_start_token is None:
            raise ValueError("inventory must start before an inventory batch is indexed")
        if batch != self.inventory_batch + 1:
            raise ValueError("inventory batches must advance in order")
        return replace(
            self,
            inventory_page_token=page_token,
            inventory_batch=batch,
        )

    def changes_indexed(self, changes_token: str) -> "GDriveSyncCheckpoint":
        """Return progress after a changes batch has been indexed."""

        if self.inventory_start_token is None:
            raise ValueError("inventory must start before changes are indexed")
        return replace(self, changes_token=changes_token)


__all__ = [
    "CHECKPOINT_SCHEMA_VERSION",
    "GDRIVE_CHECKPOINT_NAMESPACE_PREFIX",
    "GDriveSyncCheckpoint",
    "checkpoint_namespace",
]
