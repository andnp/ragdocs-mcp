"""Tests for the Google Drive to searchkernel record contract."""

from datetime import UTC, datetime

from searchkernel.domain import RecordStatus

from mcp_markdown_ragdocs.gdrive.models import DriveFile
from mcp_markdown_ragdocs.gdrive.records import map_drive_file, remote_fingerprint


def _file() -> DriveFile:
    return DriveFile(
        id="file-1",
        name="Notes",
        mime_type="text/plain",
        modified_time="2026-01-02T03:04:05Z",
        parents=("folder-1",),
        web_view_link="https://drive/file-1",
    )


def test_drive_record_has_stable_identity_and_provider_metadata() -> None:
    """
    Preserve Drive identity, MIME metadata, URI, and deterministic fingerprints.
    """
    file = _file()
    record = map_drive_file(file, workspace_id="workspace-1", body="content")

    assert record.source_kind == "gdrive"
    assert record.source_id == "file-1"
    assert record.workspace_id == "workspace-1"
    assert record.uri == "https://drive/file-1"
    assert record.body == "content"
    assert record.metadata["mime_type"] == "text/plain"
    assert record.metadata["parent_ids"] == ["folder-1"]
    assert record.metadata["remote_fingerprint"] == remote_fingerprint(file)
    assert record.metadata["local_text_hash"]


def test_drive_record_normalizes_provider_timestamp_to_utc() -> None:
    """
    Make source timestamps comparable regardless of provider timezone format.
    """
    record = map_drive_file(_file())

    assert record.created_at == datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC)
    assert record.updated_at == record.created_at


def test_drive_status_record_has_no_partial_body_and_keeps_archive_state() -> None:
    """
    Preserve a deleted Drive record as searchable metadata without stale content.
    """
    record = map_drive_file(
        _file(),
        extraction_status="tombstone",
        extraction_reason="removed",
        status=RecordStatus.ARCHIVED,
        deleted=True,
    )

    assert record.status is RecordStatus.ARCHIVED
    assert record.body == ""
    assert record.indexed_text == ""
    assert record.metadata["deleted"] is True
    assert record.metadata["extraction_status"] == "tombstone"
    assert record.metadata["extraction_reason"] == "removed"
