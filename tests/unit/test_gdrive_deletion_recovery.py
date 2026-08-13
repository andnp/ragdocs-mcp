"""Tests for Google Drive deletion and permission-loss recovery."""

from typing import Any, cast

import pytest
from searchkernel.domain import RecordStatus

from mcp_markdown_ragdocs.adapters.sources.gdrive import GoogleDriveContentSource
from mcp_markdown_ragdocs.gdrive.extraction import (
    DEFAULT_EXTRACTION_LIMITS,
    ExtractionLimits,
    ExtractionProfile,
    ExtractionResult,
    ExtractionStatus,
)
from mcp_markdown_ragdocs.gdrive.models import DriveFile


class _ApiError(RuntimeError):
    def __init__(self, status: int, reason: str | None = None) -> None:
        super().__init__(reason or f"provider status {status}")
        self.resp = type("Response", (), {"status": status})()
        self.reason = reason


class _Client:
    def __init__(self, error: BaseException) -> None:
        self.error = error

    async def download_file(self, file_id: str) -> bytes:
        del file_id
        raise self.error


def _file() -> DriveFile:
    return DriveFile("file-1", "Notes", "text/plain", modified_time="2026-01-01T00:00:00Z")


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
    return ExtractionResult(ExtractionStatus.INDEXED, "body", profile.name, profile.version)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("error", "reason"),
    [
        (_ApiError(404), "not-found"),
        (_ApiError(410), "gone"),
        (_ApiError(403, "insufficientFilePermissions"), "insufficientFilePermissions"),
    ],
)
async def test_definitive_loss_becomes_archived_tombstone(
    error: BaseException,
    reason: str,
) -> None:
    """
    Remove stale searchable content after Drive confirms record loss.
    """
    source = GoogleDriveContentSource(
        cast(Any, _Client(error)),
        workspace_id="workspace",
        extractor=cast(Any, _extractor),
    )

    record = await source.materialize_record(_file(), scope=source.scopes[0])

    assert record.status is RecordStatus.ARCHIVED
    assert record.body == ""
    assert record.indexed_text == ""
    assert record.metadata["deleted"] is True
    assert record.metadata["extraction_reason"] == reason


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error",
    [
        _ApiError(401, "invalidCredentials"),
        _ApiError(429),
        _ApiError(500),
        _ApiError(502),
        _ApiError(503),
        TimeoutError("temporary"),
    ],
)
async def test_transient_failure_stays_recoverable_active_record(error: BaseException) -> None:
    """
    Preserve an active record when a later Drive retry can recover the content.
    """
    source = GoogleDriveContentSource(
        cast(Any, _Client(error)),
        workspace_id="workspace",
        extractor=cast(Any, _extractor),
    )

    record = await source.materialize_record(_file(), scope=source.scopes[0])

    assert record.status is RecordStatus.ACTIVE
    assert record.metadata.get("deleted") is not True
    assert record.metadata["extraction_status"] == "provider-retryable"
