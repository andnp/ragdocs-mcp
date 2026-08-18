"""Tests for the Google Drive ContentSource adapter."""

import pytest
from searchkernel.domain import RecordStatus

from mcp_markdown_ragdocs.adapters.sources.gdrive import (
    DriveContentClient,
    GoogleDriveContentSource,
)
from mcp_markdown_ragdocs.gdrive.extraction import (
    DEFAULT_EXTRACTION_LIMITS,
    ExtractionLimits,
    ExtractionProfile,
    ExtractionResult,
    ExtractionStatus,
)
from mcp_markdown_ragdocs.gdrive.models import (
    DriveChange,
    DriveChangePage,
    DriveFile,
    DriveFilePage,
    DriveScope,
    DriveStartPageToken,
    DriveWatchChannel,
)


class _Client(DriveContentClient):
    def __init__(self, files: tuple[DriveFile, ...]) -> None:
        self.files = files
        self.exports: list[tuple[str, str]] = []
        self.downloads: list[str] = []

    async def list_files_page(
        self,
        scope: DriveScope,
        *,
        page_token: str | None = None,
        page_size: int = 1000,
    ) -> DriveFilePage:
        del scope, page_token, page_size
        return DriveFilePage(self.files)

    async def export_file(self, file_id: str, export_mime_type: str) -> bytes:
        self.exports.append((file_id, export_mime_type))
        return b"exported"

    async def get_start_page_token(self, scope: DriveScope) -> DriveStartPageToken:
        del scope
        raise AssertionError("content source tests do not synchronize changes")

    async def list_changes_page(
        self,
        scope: DriveScope,
        page_token: str,
        *,
        page_size: int = 1000,
    ) -> DriveChangePage:
        del scope, page_token, page_size
        raise AssertionError("content source tests do not synchronize changes")

    async def download_file(self, file_id: str) -> bytes:
        self.downloads.append(file_id)
        return b"downloaded"

    async def get_file_metadata(self, file_id: str) -> DriveFile:
        return DriveFile(file_id, "Target", "text/plain")

    async def watch_changes(
        self,
        scope: DriveScope,
        page_token: str,
        *,
        channel_id: str,
        address: str,
        token: str | None = None,
    ) -> DriveWatchChannel:
        del scope, page_token, channel_id, address, token
        raise AssertionError("content source tests do not renew watches")

    async def stop_channel(
        self,
        channel_id: str,
        resource_id: str | None = None,
    ) -> None:
        del channel_id, resource_id
        raise AssertionError("content source tests do not stop watches")


def _file(file_id: str, mime_type: str = "text/plain") -> DriveFile:
    return DriveFile(file_id, f"{file_id}.txt", mime_type, modified_time="2026-01-01T00:00:00Z")


def _extractor(
    payload: bytes,
    mime_type: str,
    *,
    profile: ExtractionProfile | None = None,
    limits: ExtractionLimits = DEFAULT_EXTRACTION_LIMITS,
) -> ExtractionResult:
    del payload, mime_type, limits
    if profile is None:
        raise AssertionError("the source should provide an extraction profile")
    return ExtractionResult(ExtractionStatus.INDEXED, f"body:{profile.name}", profile.name, profile.version)


@pytest.mark.asyncio
async def test_source_materializes_export_and_download_content() -> None:
    """
    Use the profile export MIME for native files and download binary files directly.
    """
    client = _Client((_file("doc", "application/vnd.google-apps.document"), _file("text")))
    source = GoogleDriveContentSource(client, workspace_id="workspace", extractor=_extractor)

    records = [record async for record in source.iter_records()]

    assert [record.body for record in records] == ["body:google-docs", "body:plain-text"]
    assert client.exports == [("doc", "text/plain")]
    assert client.downloads == ["text"]


@pytest.mark.asyncio
async def test_source_preserves_unsupported_files_as_status_records() -> None:
    """
    Keep unsupported files visible as metadata records without indexing content.
    """
    source = GoogleDriveContentSource(_Client((_file("image", "image/png"),)), workspace_id="workspace")

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
        client,
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
    source = GoogleDriveContentSource(_Client(()), workspace_id="workspace")

    record = source.tombstone_for_change(DriveChange("gone", True))

    assert record is not None
    assert record.source_id == "gone"
    assert record.status is RecordStatus.ARCHIVED
    assert record.body == ""
    assert record.metadata["deleted"] is True


def test_source_creates_archived_record_for_trashed_change() -> None:
    """
    Remove trashed Drive content from search without treating it as transient.
    """
    source = GoogleDriveContentSource(_Client(()), workspace_id="workspace")

    record = source.tombstone_for_change(
        DriveChange("trashed", False, DriveFile("trashed", "Notes", "text/plain", trashed=True))
    )

    assert record is not None
    assert record.status is RecordStatus.ARCHIVED
    assert record.metadata["extraction_reason"] == "trashed"
    assert record.metadata["deleted"] is True


def _known_keys(record) -> dict[str, tuple[str, str]]:
    return {
        record.source_id: (
            record.metadata["remote_fingerprint"],
            record.metadata["processing_fingerprint"],
        )
    }


@pytest.mark.asyncio
async def test_unchanged_file_is_not_refetched() -> None:
    """
    Skip the download when the change key matches the prior materialization.
    """
    client = _Client((_file("text"),))
    source = GoogleDriveContentSource(client, workspace_id="workspace", extractor=_extractor)

    first = await source.materialize_record(_file("text"), scope=None)
    second = await source.materialize_record(
        _file("text"), scope=None, known_change_keys=_known_keys(first)
    )

    assert client.downloads == ["text"]
    assert second.metadata["extraction_status"] == "unchanged"


@pytest.mark.asyncio
async def test_modified_file_is_refetched() -> None:
    """
    Never skip a file whose modified_time (and so remote_fingerprint) changed.
    """
    client = _Client((_file("text"),))
    source = GoogleDriveContentSource(client, workspace_id="workspace", extractor=_extractor)
    original = _file("text")
    modified = DriveFile(
        "text", "text.txt", "text/plain", modified_time="2026-06-01T00:00:00Z"
    )

    first = await source.materialize_record(original, scope=None)
    second = await source.materialize_record(
        modified, scope=None, known_change_keys=_known_keys(first)
    )

    assert client.downloads == ["text", "text"]
    assert second.metadata["extraction_status"] == "indexed"


@pytest.mark.asyncio
async def test_google_native_doc_without_checksum_is_skipped_when_unchanged() -> None:
    """
    Fall back to modified_time for change detection on Google-native docs.

    Docs/Sheets/Slides never carry md5Checksum/sha256Checksum, so
    remote_fingerprint must still distinguish them via modified_time alone.
    """
    doc = _file("doc", "application/vnd.google-apps.document")
    assert doc.md5_checksum is None and doc.sha256_checksum is None
    client = _Client((doc,))
    source = GoogleDriveContentSource(client, workspace_id="workspace", extractor=_extractor)

    first = await source.materialize_record(doc, scope=None)
    second = await source.materialize_record(
        doc, scope=None, known_change_keys=_known_keys(first)
    )
    modified_doc = DriveFile(
        "doc",
        "doc.txt",
        "application/vnd.google-apps.document",
        modified_time="2026-06-01T00:00:00Z",
    )
    third = await source.materialize_record(
        modified_doc, scope=None, known_change_keys=_known_keys(first)
    )

    assert client.exports == [("doc", "text/plain"), ("doc", "text/plain")]
    assert second.metadata["extraction_status"] == "unchanged"
    assert third.metadata["extraction_status"] == "indexed"


@pytest.mark.asyncio
async def test_extractor_version_bump_invalidates_cached_change_key() -> None:
    """
    Never serve a stale extraction after extractor_version changes.
    """
    client = _Client((_file("text"),))
    source = GoogleDriveContentSource(
        client, workspace_id="workspace", extractor=_extractor, extractor_version="v1"
    )

    first = await source.materialize_record(_file("text"), scope=None)
    source.extractor_version = "v2"
    second = await source.materialize_record(
        _file("text"), scope=None, known_change_keys=_known_keys(first)
    )

    assert client.downloads == ["text", "text"]
    assert second.metadata["extraction_status"] == "indexed"
