"""Tests for the Google Drive ContentSource adapter."""

from typing import Any, cast

import pytest
from searchkernel.domain import RecordStatus

from mcp_markdown_ragdocs.adapters.sources.gdrive import GoogleDriveContentSource
from mcp_markdown_ragdocs.gdrive.extraction import ExtractionResult, ExtractionStatus
from mcp_markdown_ragdocs.gdrive.models import DriveChange, DriveFile, DriveFilePage


class _Client:
    def __init__(self, files: tuple[DriveFile, ...]) -> None:
        self.files = files
        self.exports: list[tuple[str, str]] = []
        self.downloads: list[str] = []

    async def list_files_page(self, scope, *, page_token=None, page_size=1000):
        del scope, page_token, page_size
        return DriveFilePage(self.files)

    async def export_file(self, file_id: str, export_mime_type: str) -> bytes:
        self.exports.append((file_id, export_mime_type))
        return b"exported"

    async def download_file(self, file_id: str) -> bytes:
        self.downloads.append(file_id)
        return b"downloaded"

    async def get_file_metadata(self, file_id: str) -> DriveFile:
        return DriveFile(file_id, "Target", "text/plain")


def _file(file_id: str, mime_type: str = "text/plain") -> DriveFile:
    return DriveFile(file_id, f"{file_id}.txt", mime_type, modified_time="2026-01-01T00:00:00Z")


def _extractor(payload: bytes, mime_type: str, *, profile: object, limits: object) -> ExtractionResult:
    del payload, mime_type, limits
    return ExtractionResult(ExtractionStatus.INDEXED, f"body:{profile.name}", profile.name, profile.version)


@pytest.mark.asyncio
async def test_source_materializes_export_and_download_content() -> None:
    """
    Use the profile export MIME for native files and download binary files directly.
    """
    client = _Client((_file("doc", "application/vnd.google-apps.document"), _file("text")))
    source = GoogleDriveContentSource(cast(Any, client), workspace_id="workspace", extractor=_extractor)

    records = [record async for record in source.iter_records()]

    assert [record.body for record in records] == ["body:google-docs", "body:plain-text"]
    assert client.exports == [("doc", "text/plain")]
    assert client.downloads == ["text"]


@pytest.mark.asyncio
async def test_source_preserves_unsupported_files_as_status_records() -> None:
    """
    Keep unsupported files visible as metadata records without indexing content.
    """
    source = GoogleDriveContentSource(cast(Any, _Client((_file("image", "image/png"),))), workspace_id="workspace")

    record = [record async for record in source.iter_records()][0]

    assert record.body == ""
    assert record.status is RecordStatus.ACTIVE
    assert record.metadata["extraction_status"] == "unsupported"


@pytest.mark.asyncio
async def test_source_deduplicates_overlapping_scopes() -> None:
    """
    Emit one stable record when the same Drive file appears in two scopes.
    """
    client = _Client((_file("file"),))
    source = GoogleDriveContentSource(
        cast(Any, client),
        workspace_id="workspace",
        shared_drive_ids=("drive-1",),
        extractor=_extractor,
    )

    records = [record async for record in source.iter_records()]

    assert len(records) == 1
    assert records[0].metadata["scope_memberships"] == ["shared-drive:drive-1", "shared-with-me"]


def test_source_creates_archived_record_for_removed_change() -> None:
    """
    Represent a definitive removal as an archived record with no stale body.
    """
    source = GoogleDriveContentSource(cast(Any, _Client(())), workspace_id="workspace")

    record = source.tombstone_for_change(DriveChange("gone", True))

    assert record is not None
    assert record.source_id == "gone"
    assert record.status is RecordStatus.ARCHIVED
    assert record.body == ""
    assert record.metadata["deleted"] is True
