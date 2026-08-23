"""Tests for duplicate-safe Google Drive scope membership."""

from pathlib import Path
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
from mcp_markdown_ragdocs.gdrive.adapter import GDriveStateRepository
from mcp_markdown_ragdocs.gdrive.domain import GDriveScopeIdentity


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


def test_durable_scope_snapshot_survives_store_restart(tmp_path: Path) -> None:
    """
    Preserve scope visibility across a fresh membership-store instance.
    """
    repository = GDriveStateRepository(tmp_path / "gdrive-state.db")
    first = DriveScopeMembershipStore(repository)

    assert first.snapshot("workspace", "shared-with-me", ["file-1", "file-1"]) == (
        "file-1",
    )

    restarted = DriveScopeMembershipStore(
        GDriveStateRepository(tmp_path / "gdrive-state.db")
    )

    assert restarted.memberships_for("workspace", "file-1") == ("shared-with-me",)


def test_durable_scope_reconciliation_replaces_snapshot_atomically(tmp_path: Path) -> None:
    """
    Remove only IDs absent from a completed scope snapshot after a restart.
    """
    repository = GDriveStateRepository(tmp_path / "gdrive-state.db")
    identity = GDriveScopeIdentity("gdrive", "workspace", "shared-with-me")
    repository.replace_scope_memberships(identity, ("file-1", "file-2"))

    removed = repository.replace_scope_memberships(identity, ("file-2", "file-3"))
    restarted = GDriveStateRepository(tmp_path / "gdrive-state.db")

    assert removed == ("file-1",)
    assert restarted.load_scope_memberships(identity).source_ids == (
        "file-2",
        "file-3",
    )


class _EmptyClient:
    async def list_files_page(
        self,
        scope: object,
        *,
        page_token: str | None = None,
        page_size: int = 1000,
    ) -> DriveFilePage:
        del scope, page_token, page_size
        return DriveFilePage(())


@pytest.mark.asyncio
async def test_completed_empty_scope_emits_final_loss_tombstone(tmp_path: Path) -> None:
    """
    Tombstone a durable record when a complete scope snapshot no longer lists it.
    """
    repository = GDriveStateRepository(tmp_path / "gdrive-state.db")
    repository.add_membership(
        GDriveScopeIdentity("gdrive", "workspace", "shared-with-me"),
        "file-1",
    )
    source = GoogleDriveContentSource(
        cast(Any, _EmptyClient()),
        workspace_id="workspace",
        extractor=cast(Any, _extractor),
        state_repository=repository,
    )

    records = [record async for record in source.iter_records()]

    assert [record.source_id for record in records] == ["file-1"]
    assert records[0].metadata["deleted"] is True
    assert repository.memberships_for_source("gdrive", "workspace", "file-1") == ()


@pytest.mark.asyncio
async def test_scope_listing_retry_preserves_durable_membership(tmp_path: Path) -> None:
    """
    Keep the prior snapshot when a provider listing fails transiently.
    """
    repository = GDriveStateRepository(tmp_path / "gdrive-state.db")
    repository.add_membership(
        GDriveScopeIdentity("gdrive", "workspace", "shared-with-me"),
        "file-1",
    )
    source = GoogleDriveContentSource(
        cast(Any, _RetryingListClient()),
        workspace_id="workspace",
        extractor=cast(Any, _extractor),
        state_repository=repository,
    )

    with pytest.raises(_ProviderError):
        _ = [record async for record in source.iter_records()]

    assert repository.memberships_for_source("gdrive", "workspace", "file-1") == (
        "shared-with-me",
    )


@pytest.mark.parametrize(
    ("status", "reason"),
    [(403, "accessDenied"), (404, None)],
)
def test_permission_loss_removes_one_scope_before_tombstoning_record(
    status: int,
    reason: str | None,
) -> None:
    """
    Keep an overlapping record active until its final scope is lost.
    """
    membership_store = DriveScopeMembershipStore()
    source = GoogleDriveContentSource(
        cast(Any, _Client()),
        workspace_id="workspace",
        shared_drive_ids=("drive-1",),
        extractor=cast(Any, _extractor),
        membership_store=membership_store,
    )
    shared_scope, drive_scope = source.scopes
    file = DriveFile("file-1", "Notes", "text/plain")
    membership_store.add("workspace", "file-1", source.scope_identity(shared_scope))
    membership_store.add("workspace", "file-1", source.scope_identity(drive_scope))
    error = _ProviderError(status, reason)

    active = source.tombstone_for_error(file, error, scope=shared_scope)
    archived = source.tombstone_for_error(file, error, scope=drive_scope)

    assert active is not None
    assert active.metadata.get("deleted") is not True
    assert active.metadata["scope_memberships"] == ["shared-drive:drive-1"]
    assert archived is not None
    assert archived.metadata["deleted"] is True
    assert membership_store.memberships_for("workspace", "file-1") == ()


def test_retryable_provider_failure_keeps_durable_membership() -> None:
    """
    Leave scope visibility intact when a provider failure can be retried.
    """
    membership_store = DriveScopeMembershipStore()
    source = GoogleDriveContentSource(
        cast(Any, _Client()),
        workspace_id="workspace",
        extractor=cast(Any, _extractor),
        membership_store=membership_store,
    )
    scope = source.scopes[0]
    membership_store.add("workspace", "file-1", source.scope_identity(scope))

    assert source.tombstone_for_error(
        DriveFile("file-1", "Notes", "text/plain"),
        _ProviderError(503, "backendError"),
        scope=scope,
    ) is None
    assert membership_store.memberships_for("workspace", "file-1") == ("shared-with-me",)


class _ProviderError(RuntimeError):
    def __init__(self, status: int, reason: str | None) -> None:
        super().__init__(reason or str(status))
        self.resp = type("Response", (), {"status": status})()
        self.reason = reason


class _RetryingListClient:
    async def list_files_page(
        self,
        scope: object,
        *,
        page_token: str | None = None,
        page_size: int = 1000,
    ) -> DriveFilePage:
        del scope, page_token, page_size
        raise _ProviderError(503, "backendError")
