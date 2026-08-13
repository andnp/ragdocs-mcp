"""Versioned Google Drive synchronization checkpoint values."""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
from pathlib import Path

from searchkernel.api import atomic_write_json

CHECKPOINT_SCHEMA_VERSION = 1
GDRIVE_CHECKPOINT_NAMESPACE_PREFIX = "gdrive-v1"
GDRIVE_CHECKPOINT_FILENAME = "gdrive-sync-checkpoints.json"


def checkpoint_namespace(scope_generation: str) -> str:
    """Return the source-specific namespace for one Drive scope generation."""

    if not scope_generation or ":" in scope_generation:
        raise ValueError("scope_generation must be non-empty and must not contain ':'")
    return f"{GDRIVE_CHECKPOINT_NAMESPACE_PREFIX}:{scope_generation}"


def checkpoint_path(index_root: Path) -> Path:
    """Return the index-root path used for Drive synchronization state."""

    return index_root / GDRIVE_CHECKPOINT_FILENAME


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
        for field_name, value in (
            ("inventory_start_token", self.inventory_start_token),
            ("inventory_page_token", self.inventory_page_token),
            ("changes_token", self.changes_token),
        ):
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
        schema_version = payload.get("schema_version")
        inventory_start_token = payload.get("inventory_start_token")
        inventory_page_token = payload.get("inventory_page_token")
        inventory_batch = payload.get("inventory_batch")
        changes_token = payload.get("changes_token")
        if not isinstance(schema_version, int):
            raise ValueError("schema_version must be an integer")
        if not isinstance(inventory_batch, int) or isinstance(inventory_batch, bool):
            raise ValueError("inventory_batch must be a non-negative integer")
        for field_name, value in (
            ("inventory_start_token", inventory_start_token),
            ("inventory_page_token", inventory_page_token),
            ("changes_token", changes_token),
        ):
            if value is not None and not isinstance(value, str):
                raise ValueError(f"{field_name} must be a non-empty string or null")
        return cls(
            schema_version=schema_version,
            inventory_start_token=inventory_start_token,
            inventory_page_token=inventory_page_token,
            inventory_batch=inventory_batch,
            changes_token=changes_token,
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


class GDriveSyncCheckpointStore:
    """Atomically persist Drive checkpoints in the configured index root."""

    def __init__(self, index_root: Path) -> None:
        self.path = checkpoint_path(index_root)

    def load(self, namespace: str) -> GDriveSyncCheckpoint | None:
        """Load one namespace, treating missing or invalid state as empty."""

        _validate_namespace(namespace)
        payload = self._read_payload()
        raw_checkpoint = payload.get(namespace)
        if raw_checkpoint is None:
            return None
        try:
            return GDriveSyncCheckpoint.from_payload(raw_checkpoint)
        except ValueError:
            return None

    def save(self, namespace: str, checkpoint: GDriveSyncCheckpoint) -> None:
        """Atomically replace one namespace while retaining other checkpoints."""

        _validate_namespace(namespace)
        payload = self._read_payload()
        payload[namespace] = checkpoint.to_payload()
        atomic_write_json(
            self.path,
            {
                "schema_version": CHECKPOINT_SCHEMA_VERSION,
                "checkpoints": payload,
            },
        )

    def begin_inventory(
        self,
        namespace: str,
        start_token: str,
    ) -> GDriveSyncCheckpoint:
        """Persist a start token before the caller begins inventory."""

        checkpoint = (self.load(namespace) or GDriveSyncCheckpoint()).inventory_started(
            start_token
        )
        self.save(namespace, checkpoint)
        return checkpoint

    def persist_inventory_batch_after_index(
        self,
        namespace: str,
        *,
        page_token: str | None,
        batch: int,
    ) -> GDriveSyncCheckpoint:
        """Persist inventory progress after the caller commits its index mutation."""

        current = self._require(namespace)
        checkpoint = current.inventory_batch_indexed(
            page_token=page_token,
            batch=batch,
        )
        self.save(namespace, checkpoint)
        return checkpoint

    def persist_changes_after_index(
        self,
        namespace: str,
        changes_token: str,
    ) -> GDriveSyncCheckpoint:
        """Persist a changes cursor after the caller commits its index mutation."""

        current = self._require(namespace)
        checkpoint = current.changes_indexed(changes_token)
        self.save(namespace, checkpoint)
        return checkpoint

    def _require(self, namespace: str) -> GDriveSyncCheckpoint:
        checkpoint = self.load(namespace)
        if checkpoint is None:
            raise ValueError(f"no checkpoint exists for namespace {namespace!r}")
        return checkpoint

    def _read_payload(self) -> dict[str, object]:
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (FileNotFoundError, OSError, json.JSONDecodeError):
            return {}
        if not isinstance(raw, dict):
            return {}
        if raw.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
            return {}
        checkpoints = raw.get("checkpoints")
        if not isinstance(checkpoints, dict):
            return {}
        return dict(checkpoints)


def _validate_namespace(namespace: str) -> None:
    prefix = f"{GDRIVE_CHECKPOINT_NAMESPACE_PREFIX}:"
    if not namespace.startswith(prefix) or not namespace.removeprefix(prefix):
        raise ValueError(f"invalid Google Drive checkpoint namespace: {namespace!r}")


__all__ = [
    "CHECKPOINT_SCHEMA_VERSION",
    "GDRIVE_CHECKPOINT_FILENAME",
    "GDRIVE_CHECKPOINT_NAMESPACE_PREFIX",
    "GDriveSyncCheckpoint",
    "GDriveSyncCheckpointStore",
    "checkpoint_namespace",
    "checkpoint_path",
]
