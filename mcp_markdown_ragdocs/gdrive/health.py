"""Source-specific Google Drive freshness and availability health."""

from __future__ import annotations

import json
import time
from collections.abc import Sequence
from dataclasses import dataclass, replace
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

    @classmethod
    def from_payload(cls, payload: object) -> "DriveScopeHealth":
        if not isinstance(payload, dict):
            raise ValueError("Drive scope health must be an object")
        scope_identity = payload.get("scope_identity")
        if not isinstance(scope_identity, str):
            raise ValueError("scope_identity is required")
        indexed_records = payload.get("indexed_records", 0)
        remote_records = payload.get("remote_records")
        acl_complete = payload.get("acl_complete", True)
        last_success_at = payload.get("last_success_at")
        last_error = payload.get("last_error")
        if not isinstance(indexed_records, int) or isinstance(indexed_records, bool):
            raise ValueError("indexed_records must be an integer")
        if remote_records is not None and (
            not isinstance(remote_records, int) or isinstance(remote_records, bool)
        ):
            raise ValueError("remote_records must be an integer or null")
        if not isinstance(acl_complete, bool):
            raise ValueError("acl_complete must be a boolean")
        if last_success_at is not None and not isinstance(last_success_at, (int, float)):
            raise ValueError("last_success_at must be a number or null")
        if last_error is not None and not isinstance(last_error, str):
            raise ValueError("last_error must be a string or null")
        return cls(
            scope_identity,
            indexed_records=indexed_records,
            remote_records=remote_records,
            acl_complete=acl_complete,
            last_success_at=float(last_success_at) if last_success_at is not None else None,
            last_error=last_error,
        )


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
        elif any(_is_stale(scope, now, stale_after_seconds) for scope in scope_values):
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

    def scopes_for(
        self,
        workspace_id: str,
        scope_identities: Sequence[str],
        *,
        indexed_records: int = 0,
    ) -> tuple[DriveScopeHealth, ...]:
        """Load persisted scope outcomes, filling only missing scopes."""
        payload = self.load(workspace_id)
        source = payload.get("source") if isinstance(payload, dict) else None
        raw_scopes = source.get("scopes") if isinstance(source, dict) else None
        persisted: dict[str, DriveScopeHealth] = {}
        if isinstance(raw_scopes, list):
            for raw_scope in raw_scopes:
                try:
                    scope = DriveScopeHealth.from_payload(raw_scope)
                except (TypeError, ValueError):
                    continue
                persisted[scope.scope_identity] = scope
        return tuple(
            persisted.get(identity, DriveScopeHealth(identity, indexed_records=indexed_records))
            for identity in scope_identities
        )

    def record_sync_success(
        self,
        workspace_id: str,
        scope_identity: str,
        *,
        indexed_records: int,
        observed_at: float | None = None,
    ) -> None:
        self._record_sync_outcome(
            workspace_id,
            scope_identity,
            indexed_records=indexed_records,
            observed_at=observed_at,
            success=True,
            error=None,
        )

    def record_sync_failure(
        self,
        workspace_id: str,
        scope_identity: str,
        error: str,
        *,
        observed_at: float | None = None,
    ) -> None:
        self._record_sync_outcome(
            workspace_id,
            scope_identity,
            indexed_records=None,
            observed_at=observed_at,
            success=False,
            error=error,
        )

    def _record_sync_outcome(
        self,
        workspace_id: str,
        scope_identity: str,
        *,
        indexed_records: int | None,
        observed_at: float | None,
        success: bool,
        error: str | None,
    ) -> None:
        now = time.time() if observed_at is None else observed_at
        payload = self.load(workspace_id)
        source = payload.get("source") if isinstance(payload, dict) else None
        source = source if isinstance(source, dict) else {}
        identities = [scope_identity]
        raw_scopes = source.get("scopes")
        if isinstance(raw_scopes, list):
            identities.extend(
                str(raw_scope["scope_identity"])
                for raw_scope in raw_scopes
                if isinstance(raw_scope, dict) and isinstance(raw_scope.get("scope_identity"), str)
            )
        scopes = list(self.scopes_for(workspace_id, tuple(dict.fromkeys(identities))))
        updated: list[DriveScopeHealth] = []
        for scope in scopes:
            if scope.scope_identity != scope_identity:
                updated.append(scope)
                continue
            updated.append(
                replace(
                    scope,
                    indexed_records=(
                        scope.indexed_records if indexed_records is None else indexed_records
                    ),
                    last_success_at=now if success else scope.last_success_at,
                    last_error=None if success else error,
                )
            )
        available = success and all(scope.last_error is None for scope in updated)
        aggregate_error = next(
            (scope.last_error for scope in updated if scope.last_error is not None),
            None,
        )
        evaluated = DriveSourceHealth.evaluate(
            workspace_id,
            updated,
            available=available,
            observed_at=now,
            watch_mode=source.get("watch_mode") if isinstance(source.get("watch_mode"), str) else None,
            last_error=aggregate_error if aggregate_error is not None else error,
        )
        self.save(evaluated)

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


def _is_stale(scope: DriveScopeHealth, now: float, stale_after_seconds: float) -> bool:
    return scope.last_success_at is None or now - scope.last_success_at > stale_after_seconds


__all__ = [
    "DriveHealthStatus",
    "DriveScopeHealth",
    "DriveSourceHealth",
    "GDriveHealthStore",
    "HEALTH_SCHEMA_VERSION",
    "HEALTH_STATE_FILENAME",
]
