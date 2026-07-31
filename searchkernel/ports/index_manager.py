"""IndexManagerPort port: the index-manager surface SearchOrchestrator needs.

Decouples the search orchestrator from any concrete, app-specific index
manager (file-watching/daemon lifecycle owned by the app). The orchestrator
only ever calls the handful of methods captured here; a composition root can
pass its concrete IndexManager directly (it structurally satisfies this
Protocol) or a purpose-built value object.
"""

from typing import Protocol, runtime_checkable


@runtime_checkable
class IndexManagerPort(Protocol):
    """Minimal index-manager surface required by SearchOrchestrator."""

    def get_state_version(self) -> int:
        """Return the current index state version, used for cache keys."""
        ...

    def reindex_document(self, doc_id: str, reason: str | None = None) -> bool:
        """Reindex a document by id, returning whether it was reindexed."""
        ...

    def persist(self) -> None:
        """Persist all indices, retrying transient failures internally."""
        ...
