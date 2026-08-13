"""Google Drive content source adapter."""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable, Iterable
from datetime import UTC, datetime
from typing import Protocol

from searchkernel.domain import ChangeSignal, Cursor, Record, RecordStatus

from mcp_markdown_ragdocs.gdrive.client import GoogleDriveClient
from mcp_markdown_ragdocs.gdrive.errors import classify_provider_error
from mcp_markdown_ragdocs.gdrive.extraction import (
    DEFAULT_EXTRACTION_LIMITS,
    ExtractionLimits,
    ExtractionResult,
    ExtractionStatus,
    extract_content,
)
from mcp_markdown_ragdocs.gdrive.models import DriveChange, DriveFile, DriveScope
from mcp_markdown_ragdocs.gdrive.records import (
    SOURCE_KIND,
    extraction_profile,
    map_drive_file,
)

FOLDER_MIME_TYPE = "application/vnd.google-apps.folder"
SHORTCUT_MIME_TYPE = "application/vnd.google-apps.shortcut"


class DriveExtractor(Protocol):
    def __call__(
        self,
        payload: bytes,
        mime_type: str,
        *,
        profile: object | None = None,
        limits: ExtractionLimits = DEFAULT_EXTRACTION_LIMITS,
    ) -> ExtractionResult: ...


class GoogleDriveContentSource:
    """Expose Drive files as stable asynchronous searchkernel records."""

    source_kind = SOURCE_KIND

    def __init__(
        self,
        client: GoogleDriveClient,
        *,
        workspace_id: str,
        shared_drive_ids: Iterable[str] = (),
        extraction_limits: ExtractionLimits = DEFAULT_EXTRACTION_LIMITS,
        extractor: DriveExtractor = extract_content,
        clock: Callable[[], datetime] | None = None,
        page_size: int = 1000,
    ) -> None:
        if not workspace_id:
            raise ValueError("workspace_id is required")
        if page_size < 1:
            raise ValueError("page_size must be positive")
        self.client = client
        self.workspace_id = workspace_id
        self.page_size = page_size
        self.scopes = (
            DriveScope(workspace_id, include_shared_with_me=True),
            *(DriveScope(workspace_id, shared_drive_id=drive_id) for drive_id in dict.fromkeys(shared_drive_ids) if drive_id),
        )
        self.extraction_limits = extraction_limits
        self.extractor = extractor
        self.clock = clock or (lambda: datetime.now(UTC))

    @staticmethod
    def scope_identity(scope: DriveScope) -> str:
        if scope.shared_drive_id:
            return f"shared-drive:{scope.shared_drive_id}"
        if scope.include_shared_with_me:
            return "shared-with-me"
        raise ValueError("Drive scope must identify shared-with-me or a shared drive")

    async def iter_scope_files(self, scope: DriveScope) -> AsyncIterator[DriveFile]:
        page_token: str | None = None
        while True:
            page = await self.client.list_files_page(scope, page_token=page_token, page_size=self.page_size)
            for file in page.files:
                yield file
            if not page.next_page_token:
                return
            page_token = page.next_page_token

    async def iter_records(self, since: Cursor | None = None) -> AsyncIterator[Record]:
        del since
        records: dict[str, Record] = {}
        for scope in self.scopes:
            async for file in self.iter_scope_files(scope):
                record = await self.materialize_record(file, scope=scope)
                existing = records.get(record.source_id)
                if existing is not None:
                    self._add_scope_membership(existing, scope)
                    continue
                records[record.source_id] = record
                yield record

    def change_signal(self) -> ChangeSignal:
        return {"poll_interval": 3600, "source_kind": self.source_kind}

    def cursor_for(self, record: Record) -> Cursor:
        return str(record.metadata.get("remote_fingerprint") or record.source_id)

    async def materialize_record(self, file: DriveFile, *, scope: DriveScope | None = None) -> Record:
        if file.shortcut_target_id and file.mime_type == SHORTCUT_MIME_TYPE:
            try:
                target = await self.client.get_file_metadata(file.shortcut_target_id)
            except Exception as error:
                return self._status_record(file, scope, "shortcut-unresolved", str(error))
            if not target.id or target.id == file.id or target.shortcut_target_id:
                return self._status_record(file, scope, "shortcut-unresolved")
            record = await self.materialize_record(target, scope=scope)
            self._add_scope_membership(record, scope)
            return record

        profile = extraction_profile(file)
        if file.mime_type == FOLDER_MIME_TYPE:
            return self._status_record(file, scope, "folder")
        if profile is None:
            return self._status_record(file, scope, ExtractionStatus.UNSUPPORTED.value)
        try:
            payload = (
                await self.client.export_file(file.id, profile.export_mime_type)
                if profile.export_mime_type
                else await self.client.download_file(file.id)
            )
            result = self.extractor(payload, file.mime_type, profile=profile, limits=self.extraction_limits)
        except Exception as error:
            info = classify_provider_error(error)
            return self._status_record(file, scope, f"provider-{info.classification.value}", info.reason)
        if result.status is not ExtractionStatus.INDEXED or result.text is None:
            return self._status_record(file, scope, result.status.value, result.reason)
        return map_drive_file(
            file,
            workspace_id=self.workspace_id,
            body=result.text,
            extraction_status=result.status.value,
            extraction_reason=result.reason,
            scope_memberships=(self.scope_identity(scope),) if scope else (),
            clock=self.clock,
        )

    def tombstone_for_change(self, change: DriveChange, *, scope: DriveScope | None = None) -> Record | None:
        if not change.removed and not (change.file and change.file.trashed):
            return None
        file = change.file or DriveFile(change.file_id, "", "")
        return map_drive_file(
            file,
            workspace_id=self.workspace_id,
            extraction_status="tombstone",
            extraction_reason="removed" if change.removed else "trashed",
            scope_memberships=(self.scope_identity(scope),) if scope else (),
            clock=self.clock,
            status=RecordStatus.ARCHIVED,
            deleted=True,
        )

    def _status_record(
        self,
        file: DriveFile,
        scope: DriveScope | None,
        status: str,
        reason: str | None = None,
    ) -> Record:
        return map_drive_file(
            file,
            workspace_id=self.workspace_id,
            extraction_status=status,
            extraction_reason=reason,
            scope_memberships=(self.scope_identity(scope),) if scope else (),
            clock=self.clock,
        )

    def _add_scope_membership(self, record: Record, scope: DriveScope | None) -> None:
        if scope is None:
            return
        memberships = set(record.metadata.get("scope_memberships", ()))
        memberships.add(self.scope_identity(scope))
        record.metadata["scope_memberships"] = sorted(memberships)


__all__ = ["GoogleDriveContentSource"]
