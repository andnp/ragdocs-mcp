"""Source-specific Google Drive freshness and availability health."""

from __future__ import annotations

import json
import time
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from searchkernel.api import atomic_write_json

from mcp_markdown_ragdocs.gdrive.records import SOURCE_KIND

HEALTH_SCHEMA_VERSION = 1
HEALTH_STATE_FILENAME = "gdrive-health.json"


class DriveHealthStatus(StrEnum):
    HEALTHY = "healthy"
    EMPTY = "empty"
    STALE = "stale"
    ACL_INCOMPLETE = "acl-incomplete"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True, slots=True)
class DriveScopeHealth:
    """The source-owned health inputs for one Drive scope."""

    scope_identity: str
    indexed_records: int = 0
    remote_records: int | None = None
    acl_complete: bool = True
    last_success_at: float | None = None
    last_error: str | None = None

    def __post_init__(self) -> None:
        if not self.scope_identity:
            raise ValueError("scope_identity is required")
        if self.indexed_records < 0 or (
            self.remote_records is not None and self.remote_records < 0
        ):
            raise ValueError("Drive health record counts must be non-negative")

    def to_payload(self) -> dict[str, object]:
        return {
            "scope_identity": self.scope_identity,
            "indexed_records": self.indexed_records,
            "remote_records": self.remote_records,
            "acl_complete": self.acl_complete,
            "last_success_at": self.last_success_at,
            "last_error": self.last_error,
        }


@dataclass(frozen=True, slots=True)
class DriveSourceHealth:
    """Evaluated Drive health with enough data for source-aware degradation."""

    status: DriveHealthStatus
    workspace_id: str
    scopes: tuple[DriveScopeHealth, ...]
    available: bool
    observed_at: float
    stale_after_seconds: float
    source_kind: str = SOURCE_KIND
    watch_mode: str | None = None
    last_error: str | None = None

    @classmethod
    def evaluate(
        cls,
        workspace_id: str,
        scopes: Sequence[DriveScopeHealth],
        *,
        available: bool = True,
        observed_at: float | None = None,
        stale_after_seconds: float = 3600.0,
        watch_mode: str | None = None,
        last_error: str | None = None,
    ) -> "DriveSourceHealth":
        if not workspace_id:
            raise ValueError("workspace_id is required")
        if stale_after_seconds <= 0:
            raise ValueError("stale_after_seconds must be positive")
        scope_values = tuple(scopes)
        now = time.time() if observed_at is None else observed_at
        if not available:
            status = DriveHealthStatus.UNAVAILABLE
        elif any(not scope.acl_complete for scope in scope_values):
            status = DriveHealthStatus.ACL_INCOMPLETE
        elif _is_empty(scope_values):
            status = DriveHealthStatus.EMPTY
        elif any(
            scope.last_success_at is None
            or now - scope.last_success_at > stale_after_seconds
            for scope in scope_values
        ):
            status = DriveHealthStatus.STALE
        else:
            status = DriveHealthStatus.HEALTHY
        return cls(
            status,
            workspace_id,
            scope_values,
            available,
            now,
            stale_after_seconds,
            watch_mode=watch_mode,
            last_error=last_error,
        )

    @property
    def indexed_records(self) -> int:
        return sum(scope.indexed_records for scope in self.scopes)

    @property
    def remote_records(self) -> int | None:
        if any(scope.remote_records is None for scope in self.scopes):
            return None
        return sum(scope.remote_records or 0 for scope in self.scopes)

    @property
    def acl_incomplete_scopes(self) -> tuple[str, ...]:
        return tuple(scope.scope_identity for scope in self.scopes if not scope.acl_complete)

    @property
    def last_success_at(self) -> float | None:
        values = [scope.last_success_at for scope in self.scopes if scope.last_success_at is not None]
        return max(values) if values else None

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": HEALTH_SCHEMA_VERSION,
            "status": self.status.value,
            "source": {
                "source_kind": self.source_kind,
                "workspace_id": self.workspace_id,
                "available": self.available,
                "observed_at": self.observed_at,
                "stale_after_seconds": self.stale_after_seconds,
                "watch_mode": self.watch_mode,
                "last_error": self.last_error,
                "indexed_records": self.indexed_records,
                "remote_records": self.remote_records,
                "acl_incomplete_scopes": list(self.acl_incomplete_scopes),
                "scopes": [scope.to_payload() for scope in self.scopes],
            },
        }


class GDriveHealthStore:
    """Atomically retain the latest evaluated health for each workspace."""

    def __init__(self, index_root: Path) -> None:
        self.path = Path(index_root) / HEALTH_STATE_FILENAME

    def save(self, health: DriveSourceHealth) -> None:
        states = self._read()
        states[health.workspace_id] = health.to_payload()
        atomic_write_json(
            self.path,
            {"schema_version": HEALTH_SCHEMA_VERSION, "states": states},
        )

    def load(self, workspace_id: str) -> dict[str, object] | None:
        return self._read().get(workspace_id)

    def _read(self) -> dict[str, dict[str, object]]:
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (FileNotFoundError, OSError, json.JSONDecodeError):
            return {}
        if not isinstance(payload, dict) or payload.get("schema_version") != HEALTH_SCHEMA_VERSION:
            return {}
        states = payload.get("states")
        if not isinstance(states, dict):
            return {}
        return {
            str(workspace_id): state
            for workspace_id, state in states.items()
            if isinstance(state, dict)
        }


def _is_empty(scopes: Sequence[DriveScopeHealth]) -> bool:
    return not scopes or all(
        scope.indexed_records == 0
        and (scope.remote_records is None or scope.remote_records == 0)
        for scope in scopes
    )


__all__ = [
    "DriveHealthStatus",
    "DriveScopeHealth",
    "DriveSourceHealth",
    "GDriveHealthStore",
    "HEALTH_SCHEMA_VERSION",
    "HEALTH_STATE_FILENAME",
]
