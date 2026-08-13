"""Scope visibility state kept separate from Google Drive record identity."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class DriveScopeMembership:
    """One visibility relationship for a stable Drive record."""

    workspace_id: str
    source_id: str
    scope_identity: str


class DriveScopeMembershipStore:
    """Collect unique scope relationships without changing record identity."""

    def __init__(self) -> None:
        self._memberships: dict[tuple[str, str], set[str]] = {}

    def add(self, workspace_id: str, source_id: str, scope_identity: str) -> tuple[str, ...]:
        key = (workspace_id, source_id)
        self._memberships.setdefault(key, set()).add(scope_identity)
        return self.memberships_for(workspace_id, source_id)

    def memberships_for(self, workspace_id: str, source_id: str) -> tuple[str, ...]:
        return tuple(sorted(self._memberships.get((workspace_id, source_id), ())))

    def discard(self, workspace_id: str, source_id: str, scope_identity: str) -> tuple[str, ...]:
        key = (workspace_id, source_id)
        memberships = self._memberships.get(key)
        if memberships is None:
            return ()
        memberships.discard(scope_identity)
        if not memberships:
            self._memberships.pop(key)
            return ()
        return tuple(sorted(memberships))


__all__ = ["DriveScopeMembership", "DriveScopeMembershipStore"]
