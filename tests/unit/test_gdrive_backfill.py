"""Tests for bounded Google Drive backfill recovery."""

from pathlib import Path
from collections.abc import Sequence
from typing import Any, cast

import pytest
from searchkernel.api import Record

from mcp_markdown_ragdocs.adapters.sources.gdrive import GoogleDriveContentSource
from mcp_markdown_ragdocs.gdrive.backfill import (
    GDriveBackfillCheckpointStore,
    GoogleDriveBackfill,
)
from mcp_markdown_ragdocs.gdrive.extraction import (
    DEFAULT_EXTRACTION_LIMITS,
    ExtractionLimits,
    ExtractionProfile,
    ExtractionResult,
    ExtractionStatus,
)
from mcp_markdown_ragdocs.gdrive.models import DriveFile, DriveFilePage
from mcp_markdown_ragdocs.gdrive.records import map_drive_file
from mcp_markdown_ragdocs.gdrive.state import GDriveScopeIdentity, GDriveStateRepository


def _file(file_id: str) -> DriveFile:
    return DriveFile(file_id, f"{file_id}.txt", "text/plain")


def _extractor(
    payload: bytes,
    mime_type: str,
    *,
    profile: ExtractionProfile | None = None,
    limits: ExtractionLimits = DEFAULT_EXTRACTION_LIMITS,
) -> ExtractionResult:
    del payload, mime_type, limits
    assert profile is not None
    return ExtractionResult(
        ExtractionStatus.INDEXED,
        f"body:{profile.name}",
        profile.name,
        profile.version,
    )


class _Client:
    def __init__(self) -> None:
        self.page_tokens: list[str | None] = []
        self.pages = {
            None: DriveFilePage((_file("fresh"), _file("retry")), "page-2"),
            "page-2": DriveFilePage((_file("version"), _file("missing"))),
        }

    async def list_files_page(
        self,
        scope: object,
        *,
        page_token: str | None = None,
        page_size: int = 1000,
    ) -> DriveFilePage:
        del scope, page_size
        self.page_tokens.append(page_token)
        return self.pages[page_token]

    async def download_file(self, file_id: str) -> bytes:
        return file_id.encode()


class _Writer:
    def __init__(self) -> None:
        self.records: list[Record] = []
        self.persist_count = 0

    def index_records(self, records: Sequence[Record]) -> bool:
        self.records.extend(records)
        return True

    def persist(self) -> None:
        self.persist_count += 1


@pytest.mark.asyncio
async def test_backfill_reprocesses_failed_missing_and_versioned_records(
    tmp_path: Path,
) -> None:
    """
    Reprocess only records that need recovery or a new processing fingerprint.
    """
    client = _Client()
    source = GoogleDriveContentSource(
        cast(Any, client),
        workspace_id="workspace",
        extractor=cast(Any, _extractor),
        extractor_version="v2",
        chunker_version="v3",
    )
    existing = {
        "fresh": map_drive_file(
            _file("fresh"), extractor_version="v2", chunker_version="v3"
        ),
        "retry": map_drive_file(
            _file("retry"),
            extraction_status="provider-retryable",
            extractor_version="v2",
            chunker_version="v3",
        ),
        "version": map_drive_file(_file("version")),
    }
    writer = _Writer()

    progress = await GoogleDriveBackfill(
        source,
        GDriveBackfillCheckpointStore(tmp_path),
        writer,
        scope_generation="generation",
        max_seconds=60,
    ).run(source.scopes[0], existing)

    assert progress.complete is True
    assert [record.source_id for record in writer.records] == [
        "retry",
        "version",
        "missing",
    ]
    assert writer.persist_count == 2


@pytest.mark.asyncio
async def test_backfill_resumes_from_checkpointed_page_bound(tmp_path: Path) -> None:
    """
    Resume the next inventory page after a bounded backfill pass.
    """
    client = _Client()
    source = GoogleDriveContentSource(
        cast(Any, client),
        workspace_id="workspace",
        extractor=cast(Any, _extractor),
    )
    store = GDriveBackfillCheckpointStore(tmp_path)
    writer = _Writer()
    backfill = GoogleDriveBackfill(
        source,
        store,
        writer,
        scope_generation="generation",
        max_pages=1,
        max_seconds=60,
    )

    first = await backfill.run(source.scopes[0], {})
    second = await backfill.run(source.scopes[0], {})

    assert first.complete is False
    assert second.complete is True
    assert client.page_tokens == [None, "page-2"]
    assert first.items_scanned == 2
    assert second.items_scanned == 2
    assert writer.persist_count == 2


@pytest.mark.asyncio
async def test_backfill_persists_each_configured_record_batch(tmp_path: Path) -> None:
    """
    Split reprocessed records into bounded durable index writes.
    """
    client = _Client()
    client.pages = {None: DriveFilePage((_file("first"), _file("second")))}
    source = GoogleDriveContentSource(
        cast(Any, client),
        workspace_id="workspace",
        extractor=cast(Any, _extractor),
    )
    writer = _Writer()

    progress = await GoogleDriveBackfill(
        source,
        GDriveBackfillCheckpointStore(tmp_path),
        writer,
        scope_generation="generation",
        batch_size=1,
        max_seconds=60,
    ).run(source.scopes[0], {})

    assert progress.complete is True
    assert [record.source_id for record in writer.records] == ["first", "second"]
    assert writer.persist_count == 2


@pytest.mark.asyncio
async def test_complete_backfill_replaces_stale_scope_memberships(tmp_path: Path) -> None:
    """
    Reconcile a scope snapshot after every listed page has been processed.
    """
    client = _Client()
    repository = GDriveStateRepository(tmp_path / "gdrive-state.db")
    identity = GDriveScopeIdentity("gdrive", "workspace", "shared-with-me")
    repository.add_membership(identity, "stale")
    source = GoogleDriveContentSource(
        cast(Any, client),
        workspace_id="workspace",
        extractor=cast(Any, _extractor),
        state_repository=repository,
    )
    writer = _Writer()

    progress = await GoogleDriveBackfill(
        source,
        GDriveBackfillCheckpointStore(tmp_path),
        writer,
        scope_generation="generation",
        max_seconds=60,
    ).run(source.scopes[0], {})

    assert progress.complete is True
    assert repository.load_scope_memberships(identity).source_ids == (
        "fresh",
        "missing",
        "retry",
        "version",
    )
