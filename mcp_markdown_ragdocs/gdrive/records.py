"""Provider-neutral record mapping for Google Drive files."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from searchkernel.domain import Record, RecordStatus

from mcp_markdown_ragdocs.gdrive.extraction import (
    EXTRACTION_PROFILES,
    ExtractionProfile,
)
from mcp_markdown_ragdocs.gdrive.models import DriveFile

SOURCE_KIND = "gdrive"
RECORD_SCHEMA_VERSION = "gdrive-record-v1"

_MIME_PROFILE_NAMES = {
    "application/vnd.google-apps.document": "google-docs",
    "application/vnd.google-apps.spreadsheet": "google-sheets",
    "application/vnd.google-apps.presentation": "google-slides",
    "application/pdf": "pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": "xlsx",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation": "pptx",
    "text/plain": "plain-text",
    "text/markdown": "markdown",
    "text/x-markdown": "markdown",
}


def _digest(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def modified_datetime(value: str | None, fallback: Callable[[], datetime]) -> datetime:
    if value:
        try:
            parsed = datetime.fromisoformat(value)
        except ValueError:
            pass
        else:
            return parsed.replace(tzinfo=UTC) if parsed.tzinfo is None else parsed.astimezone(UTC)
    current = fallback()
    return current.replace(tzinfo=UTC) if current.tzinfo is None else current.astimezone(UTC)


def extraction_profile(file: DriveFile) -> ExtractionProfile | None:
    return EXTRACTION_PROFILES.get(_MIME_PROFILE_NAMES.get(file.mime_type, ""))


def remote_fingerprint(file: DriveFile) -> str:
    return _digest(
        {
            "id": file.id,
            "mime_type": file.mime_type,
            "modified_time": file.modified_time,
            "size": file.size,
            "md5_checksum": file.md5_checksum,
            "sha256_checksum": file.sha256_checksum,
            "drive_id": file.drive_id,
            "trashed": file.trashed,
            "shortcut_target_id": file.shortcut_target_id,
            "shortcut_target_mime_type": file.shortcut_target_mime_type,
        }
    )


def processing_fingerprint(
    file: DriveFile,
    profile: ExtractionProfile | None = None,
    *,
    extractor_version: str = "v1",
    chunker_version: str = "v1",
) -> str:
    selected = profile or extraction_profile(file)
    return _digest(
        {
            "export_mime_type": selected.export_mime_type if selected else None,
            "extractor": selected.extractor if selected else None,
            "extractor_profile": selected.name if selected else None,
            "extractor_profile_version": selected.version if selected else None,
            "extractor_version": extractor_version,
            "chunker_version": chunker_version,
            "record_schema_version": RECORD_SCHEMA_VERSION,
            "mime_type": file.mime_type,
        }
    )


def map_drive_file(
    file: DriveFile,
    *,
    workspace_id: str | None = None,
    body: str = "",
    extraction_status: str = "metadata-only",
    extraction_reason: str | None = None,
    scope_memberships: tuple[str, ...] = (),
    clock: Callable[[], datetime] | None = None,
    status: RecordStatus = RecordStatus.ACTIVE,
    deleted: bool = False,
    extractor_version: str = "v1",
    chunker_version: str = "v1",
) -> Record:
    """Map one Drive file to a stable, provider-neutral Record."""
    profile = extraction_profile(file)
    timestamp = modified_datetime(file.modified_time, clock or (lambda: datetime.now(UTC)))
    metadata: dict[str, Any] = {
        "name": file.name,
        "mime_type": file.mime_type,
        "drive_id": file.drive_id,
        "parent_ids": list(file.parents),
        "modified_time": file.modified_time,
        "scope_memberships": list(scope_memberships),
        "remote_fingerprint": remote_fingerprint(file),
        "processing_fingerprint": processing_fingerprint(
            file,
            profile,
            extractor_version=extractor_version,
            chunker_version=chunker_version,
        ),
        "extraction_profile": profile.name if profile else None,
        "extraction_profile_version": profile.version if profile else None,
        "extraction_status": extraction_status,
        "extraction_reason": extraction_reason,
    }
    if body:
        metadata["local_text_hash"] = hashlib.sha256(body.encode()).hexdigest()
    if deleted:
        metadata["deleted"] = True
    return Record(
        source_kind=SOURCE_KIND,
        source_id=file.id,
        workspace_id=workspace_id,
        title=file.name,
        body=body,
        indexed_text=body,
        created_at=timestamp,
        updated_at=timestamp,
        metadata=metadata,
        uri=file.web_view_link,
        status=status,
    )


__all__ = [
    "RECORD_SCHEMA_VERSION",
    "SOURCE_KIND",
    "extraction_profile",
    "map_drive_file",
    "modified_datetime",
    "processing_fingerprint",
    "remote_fingerprint",
]
