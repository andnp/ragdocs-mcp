"""Tests for duplicate-safe Google Drive scope membership."""

from typing import Any, cast

import pytest

from mcp_markdown_ragdocs.adapters.sources.gdrive import GoogleDriveContentSource
from mcp_markdown_ragdocs.gdrive.extraction import (
    DEFAULT_EXTRACTION_LIMITS,
    ExtractionLimits,
    ExtractionProfile,
    ExtractionResult,
    ExtractionStatus,
)
from mcp_markdown_ragdocs.gdrive.membership import DriveScopeMembershipStore
from mcp_markdown_ragdocs.gdrive.models import DriveFile, DriveFilePage


class _Client:
    async def list_files_page(
        self,
        scope: object,
        *,
        page_token: str | None = None,
        page_size: int = 1000,
    ) -> DriveFilePage:
        del scope, page_token, page_size
        return DriveFilePage((DriveFile("file-1", "Notes", "text/plain"),))

    async def download_file(self, file_id: str) -> bytes:
        return file_id.encode()


def _extractor(
    payload: bytes,
    mime_type: str,
    *,
    profile: ExtractionProfile | None = None,
    limits: ExtractionLimits = DEFAULT_EXTRACTION_LIMITS,
) -> ExtractionResult:
    del payload, mime_type, limits
    if profile is None:
        raise ValueError("profile is required")
    return ExtractionResult(ExtractionStatus.INDEXED, profile.name, profile.name, profile.version)


def test_membership_store_is_set_like() -> None:
    """
    Keep repeated provider observations from creating duplicate relationships.
    """
    store = DriveScopeMembershipStore()

    assert store.add("workspace", "file-1", "shared-with-me") == ("shared-with-me",)
    assert store.add("workspace", "file-1", "shared-with-me") == ("shared-with-me",)
    assert store.add("workspace", "file-1", "shared-drive:drive-1") == (
        "shared-drive:drive-1",
        "shared-with-me",
    )


@pytest.mark.asyncio
async def test_overlapping_scopes_emit_one_record_with_unique_memberships() -> None:
    """
    Deduplicate embeddings by stable source ID while retaining all visibility scopes.
    """
    membership_store = DriveScopeMembershipStore()
    source = GoogleDriveContentSource(
        cast(Any, _Client()),
        workspace_id="workspace",
        shared_drive_ids=("drive-1", "drive-1"),
        extractor=cast(Any, _extractor),
        membership_store=membership_store,
    )

    records = [record async for record in source.iter_records()]

    assert len(records) == 1
    assert records[0].source_id == "file-1"
    assert records[0].metadata["scope_memberships"] == [
        "shared-drive:drive-1",
        "shared-with-me",
    ]
    assert membership_store.memberships_for("workspace", "file-1") == (
        "shared-drive:drive-1",
        "shared-with-me",
    )
