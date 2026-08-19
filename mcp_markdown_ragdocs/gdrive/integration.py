"""Compose the Google Drive replacement collaborators behind one port."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

from searchkernel.api import ContentSource, Record, SemanticRecordIngestor

from mcp_markdown_ragdocs.gdrive.adapter import GDriveStateRepository
from mcp_markdown_ragdocs.gdrive.records import SOURCE_KIND
from mcp_markdown_ragdocs.gdrive.replacement import (
    GDriveReplacementJournal,
    REPLACEMENT_JOURNAL_FILENAME,
)
from mcp_markdown_ragdocs.gdrive.replacement_policy import GDriveReplacementPolicy
from mcp_markdown_ragdocs.indexing.record_ports import RecordStorage, SourceMapStore


class GDriveIntegration:
    """Adapt the replacement policy to the record manager's gdrive port."""

    source_kind = SOURCE_KIND

    def __init__(self, policy: GDriveReplacementPolicy) -> None:
        self._policy = policy

    async def replace(self, records: Sequence[Record]) -> None:
        await self._policy.replace(records)

    def recover(self) -> bool:
        return self._policy.recover()


def build_gdrive_integration(
    index_path: Path,
    content_sources: Mapping[str, ContentSource],
    ingestor: SemanticRecordIngestor,
    storage: RecordStorage,
    source_records: dict[str, list[str]],
    source_map_store: SourceMapStore,
) -> GDriveIntegration:
    """Build the gdrive replacement collaborators the record manager needs."""

    state_repository = (
        GDriveStateRepository(index_path / "gdrive-state.db")
        if SOURCE_KIND in content_sources
        else None
    )
    journal = GDriveReplacementJournal(
        index_path / REPLACEMENT_JOURNAL_FILENAME,
        state_repository,
    )
    policy = GDriveReplacementPolicy(
        ingestor,
        storage,
        source_records,
        source_map_store,
        journal,
        state_repository,
    )
    return GDriveIntegration(policy)


__all__ = ["GDriveIntegration", "build_gdrive_integration"]
