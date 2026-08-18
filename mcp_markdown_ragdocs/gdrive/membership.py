"""Scope visibility state kept separate from Google Drive record identity."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from mcp_markdown_ragdocs.gdrive.records import SOURCE_KIND
from mcp_markdown_ragdocs.gdrive.state import GDriveScopeIdentity, GDriveStatePort


@dataclass(frozen=True, slots=True)
class DriveScopeMembership:
    """One visibility relationship for a stable Drive record."""

    workspace_id: str
    source_id: str
    scope_identity: str


class DriveScopeMembershipStore:
    """Keep unique scope relationships in memory or durable Drive state."""

    def __init__(self, state_repository: GDriveStatePort | None = None) -> None:
        self._memberships: dict[tuple[str, str], set[str]] = {}
        self._scope_memberships: dict[tuple[str, str], set[str]] = {}
        self._state_repository = state_repository

    @property
    def is_durable(self) -> bool:
        """Whether membership mutations are persisted across process restarts."""

        return self._state_repository is not None

    def snapshot(
        self,
        workspace_id: str,
        scope_identity: str,
        source_ids: Iterable[str],
    ) -> tuple[str, ...]:
        """Replace one complete scope observation and return its source IDs."""

        observed = tuple(sorted(set(source_ids)))
        self.reconcile_scope(workspace_id, scope_identity, observed)
        return observed

    def add(self, workspace_id: str, source_id: str, scope_identity: str) -> tuple[str, ...]:
        if self._state_repository is not None:
            memberships = self._state_repository.add_membership(
                self._identity(workspace_id, scope_identity), source_id
            )
            self._remember(workspace_id, source_id, scope_identity)
            return memberships
        self._remember(workspace_id, source_id, scope_identity)
        return self.memberships_for(workspace_id, source_id)

    def memberships_for(self, workspace_id: str, source_id: str) -> tuple[str, ...]:
        if self._state_repository is not None:
            return self._state_repository.memberships_for_source(
                SOURCE_KIND, workspace_id, source_id
            )
        return tuple(sorted(self._memberships.get((workspace_id, source_id), ())))

    def discard(self, workspace_id: str, source_id: str, scope_identity: str) -> tuple[str, ...]:
        if self._state_repository is not None:
            remaining = self._state_repository.remove_membership(
                self._identity(workspace_id, scope_identity), source_id
            )
            self._forget(workspace_id, source_id, scope_identity)
            return remaining
        key = (workspace_id, source_id)
        memberships = self._memberships.get(key)
        if memberships is None:
            return ()
        memberships.discard(scope_identity)
        if not memberships:
            self._memberships.pop(key)
        self._forget(workspace_id, source_id, scope_identity)
        return tuple(sorted(memberships))

    def source_ids_for_scope(self, workspace_id: str, scope_identity: str) -> tuple[str, ...]:
        """Return the durable or in-memory source IDs in one scope."""

        if self._state_repository is not None:
            return self._state_repository.load_scope_memberships(
                self._identity(workspace_id, scope_identity)
            ).source_ids
        return tuple(sorted(self._scope_memberships.get((workspace_id, scope_identity), ())))

    def reconcile_scope(
        self,
        workspace_id: str,
        scope_identity: str,
        source_ids: Iterable[str],
    ) -> tuple[str, ...]:
        """Replace one completed scope snapshot and return removed source IDs."""

        normalized = tuple(sorted(set(source_ids)))
        if self._state_repository is not None:
            removed = self._state_repository.replace_scope_memberships(
                self._identity(workspace_id, scope_identity), normalized
            )
        else:
            scope_key = (workspace_id, scope_identity)
            previous = self._scope_memberships.get(scope_key, set())
            removed = tuple(sorted(previous.difference(normalized)))
            self._scope_memberships[scope_key] = set(normalized)
        for source_id in removed:
            self._forget(workspace_id, source_id, scope_identity)
        for source_id in normalized:
            self._remember(workspace_id, source_id, scope_identity)
        return removed

    def _remember(self, workspace_id: str, source_id: str, scope_identity: str) -> None:
        self._memberships.setdefault((workspace_id, source_id), set()).add(scope_identity)
        self._scope_memberships.setdefault((workspace_id, scope_identity), set()).add(source_id)

    def _forget(self, workspace_id: str, source_id: str, scope_identity: str) -> None:
        memberships = self._memberships.get((workspace_id, source_id))
        if memberships is not None:
            memberships.discard(scope_identity)
            if not memberships:
                self._memberships.pop((workspace_id, source_id))
        scope_memberships = self._scope_memberships.get((workspace_id, scope_identity))
        if scope_memberships is not None:
            scope_memberships.discard(source_id)
            if not scope_memberships:
                self._scope_memberships.pop((workspace_id, scope_identity))

    @staticmethod
    def _identity(workspace_id: str, scope_identity: str) -> GDriveScopeIdentity:
        return GDriveScopeIdentity(SOURCE_KIND, workspace_id, scope_identity)


__all__ = ["DriveScopeMembership", "DriveScopeMembershipStore"]
