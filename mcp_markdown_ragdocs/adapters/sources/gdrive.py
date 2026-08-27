"""Google Drive content source adapter."""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable, Iterable, Mapping
from datetime import UTC, datetime
from typing import Protocol

from searchkernel.domain import ChangeSignal, Cursor, Record, RecordStatus

from mcp_markdown_ragdocs.gdrive.errors import (
    ProviderErrorClassification,
    classify_provider_error,
)
from mcp_markdown_ragdocs.gdrive.extraction import (
    DEFAULT_EXTRACTION_LIMITS,
    ExtractionLimits,
    ExtractionProfile,
    ExtractionResult,
    ExtractionStatus,
    extract_content,
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
from mcp_markdown_ragdocs.gdrive.membership import DriveScopeMembershipStore
from mcp_markdown_ragdocs.gdrive.records import (
    SOURCE_KIND,
    extraction_profile,
    map_drive_file,
    processing_fingerprint,
    remote_fingerprint,
)
from mcp_markdown_ragdocs.gdrive.retry import DriveRetryWorkStore
from mcp_markdown_ragdocs.gdrive.port import GDriveStatePort

FOLDER_MIME_TYPE = "application/vnd.google-apps.folder"
SHORTCUT_MIME_TYPE = "application/vnd.google-apps.shortcut"
UNCHANGED_STATUS = "unchanged"
DEFINITIVE_PROVIDER_STATUS = f"provider-{ProviderErrorClassification.DEFINITIVE.value}"
# Outcomes that stay the same until the file or the processing versions change,
# so the change key that produced them is safe to remember. Both fingerprints
# are part of that key, so a new extractor or chunker version retries on its
# own. Retryable failures are absent on purpose: they say nothing durable.
DETERMINISTIC_MATERIALIZATION_STATUSES = frozenset(
    {
        ExtractionStatus.INDEXED.value,
        ExtractionStatus.UNSUPPORTED.value,
        ExtractionStatus.TOO_LARGE.value,
        ExtractionStatus.TRUNCATED.value,
        DEFINITIVE_PROVIDER_STATUS,
    }
)


class DriveExtractor(Protocol):
    def __call__(
        self,
        payload: bytes,
        mime_type: str,
        *,
        profile: ExtractionProfile | None = None,
        limits: ExtractionLimits = DEFAULT_EXTRACTION_LIMITS,
    ) -> ExtractionResult: ...


class DriveContentClient(Protocol):
    async def list_files_page(
        self,
        scope: DriveScope,
        *,
        page_token: str | None = None,
        page_size: int = 1000,
    ) -> DriveFilePage: ...

    async def get_start_page_token(self, scope: DriveScope) -> DriveStartPageToken: ...

    async def list_changes_page(
        self,
        scope: DriveScope,
        page_token: str,
        *,
        page_size: int = 1000,
    ) -> DriveChangePage: ...

    async def export_file(self, file_id: str, export_mime_type: str) -> bytes: ...

    async def download_file(self, file_id: str) -> bytes: ...

    async def get_file_metadata(self, file_id: str) -> DriveFile: ...

    async def watch_changes(
        self,
        scope: DriveScope,
        page_token: str,
        *,
        channel_id: str,
        address: str,
        token: str | None = None,
    ) -> DriveWatchChannel: ...

    async def stop_channel(
        self,
        channel_id: str,
        resource_id: str | None = None,
    ) -> None: ...


class GoogleDriveContentSource:
    """Expose Drive files as stable asynchronous searchkernel records."""

    source_kind = SOURCE_KIND

    def __init__(
        self,
        client: DriveContentClient,
        *,
        workspace_id: str,
        shared_drive_ids: Iterable[str] = (),
        extraction_limits: ExtractionLimits = DEFAULT_EXTRACTION_LIMITS,
        extractor: DriveExtractor = extract_content,
        clock: Callable[[], datetime] | None = None,
        page_size: int = 1000,
        retry_work_store: DriveRetryWorkStore | None = None,
        membership_store: DriveScopeMembershipStore | None = None,
        state_repository: GDriveStatePort | None = None,
        extractor_version: str = "v1",
        chunker_version: str = "v1",
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
        self.retry_work_store = retry_work_store
        self.membership_store = membership_store or DriveScopeMembershipStore(state_repository)
        self.extractor_version = extractor_version
        self.chunker_version = chunker_version

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

    async def collect_scope_source_ids(self, scope: DriveScope) -> tuple[str, ...]:
        """Materialize a complete scope pass for restart-safe reconciliation."""

        source_ids: set[str] = set()
        async for file in self.iter_scope_files(scope):
            record = await self.materialize_record(file, scope=scope)
            if self._record_visible_in_scope(record, scope):
                source_ids.add(record.source_id)
        return tuple(sorted(source_ids))

    def reconcile_scope(self, scope: DriveScope, source_ids: Iterable[str]) -> tuple[Record, ...]:
        """Replace a completed scope snapshot and build final-loss tombstones."""

        source_ids = tuple(source_ids)
        tombstones = self.scope_loss_tombstones(scope, source_ids)
        self.membership_store.reconcile_scope(
            self.workspace_id,
            self.scope_identity(scope),
            source_ids,
        )
        return tombstones

    def scope_loss_tombstones(
        self,
        scope: DriveScope,
        source_ids: Iterable[str],
    ) -> tuple[Record, ...]:
        """Build tombstones for records losing their final scope membership."""

        expected = set(source_ids)
        scope_identity = self.scope_identity(scope)
        removed = set(self.membership_store.source_ids_for_scope(
            self.workspace_id,
            scope_identity,
        )).difference(expected)
        tombstones: list[Record] = []
        for source_id in removed:
            remaining = set(self.membership_store.memberships_for(self.workspace_id, source_id))
            remaining.discard(scope_identity)
            if not remaining:
                tombstone = self.tombstone_for_change(DriveChange(source_id, True))
                if tombstone is not None:
                    tombstones.append(tombstone)
        return tuple(tombstones)

    async def iter_records(self, since: Cursor | None = None) -> AsyncIterator[Record]:
        del since
        records: dict[str, Record] = {}
        for scope in self.scopes:
            observed: set[str] = set()
            try:
                async for file in self.iter_scope_files(scope):
                    record = await self.materialize_record(file, scope=scope)
                    self._add_scope_membership(record, scope)
                    if self._record_visible_in_scope(record, scope):
                        observed.add(record.source_id)
                    existing = records.get(record.source_id)
                    if existing is not None:
                        self._add_scope_membership(existing, scope)
                        continue
                    records[record.source_id] = record
                    yield record
            except Exception as error:
                if not classify_provider_error(error).tombstone:
                    raise
                observed.clear()
            for tombstone in self.reconcile_scope(scope, observed):
                existing = records.get(tombstone.source_id)
                if existing is None or existing.metadata.get("deleted") is not True:
                    yield tombstone

    def change_signal(self) -> ChangeSignal:
        return {"poll_interval": 3600, "source_kind": self.source_kind}

    def cursor_for(self, record: Record) -> Cursor:
        return str(record.metadata.get("remote_fingerprint") or record.source_id)

    async def materialize_record(
        self,
        file: DriveFile,
        *,
        scope: DriveScope | None = None,
        known_change_keys: Mapping[str, tuple[str, str]] | None = None,
    ) -> Record:
        """Map one Drive file to a stable, provider-neutral Record.

        ``known_change_keys`` maps ``source_id`` to the (remote_fingerprint,
        processing_fingerprint) pair last durably indexed for that file. When
        the file's freshly computed pair matches, the expensive export or
        download is skipped: the returned record carries
        ``extraction_status == UNCHANGED_STATUS`` so the caller can reuse the
        already-indexed record instead of re-indexing this placeholder.
        Callers that omit ``known_change_keys`` get the previous
        unconditional-fetch behavior. Shortcuts are excluded from this check:
        the returned record is keyed by the shortcut target's id, not the
        shortcut's own id, so a caller cannot safely infer scope membership
        from a skip signalled at this level.
        """
        if file.shortcut_target_id and file.mime_type == SHORTCUT_MIME_TYPE:
            try:
                target = await self.client.get_file_metadata(file.shortcut_target_id)
            except Exception as error:
                tombstone = self.tombstone_for_error(file, error, scope=scope)
                if tombstone is not None:
                    return tombstone
                return self._status_record(file, scope, "shortcut-unresolved", str(error))
            if not target.id or target.id == file.id or target.shortcut_target_id:
                return self._status_record(file, scope, "shortcut-unresolved")
            record = await self.materialize_record(target, scope=scope)
            self._add_scope_membership(record, scope)
            return record

        self._add_scope_membership_for_source(file.id, scope)
        profile = extraction_profile(file)
        if file.mime_type == FOLDER_MIME_TYPE:
            return self._status_record(file, scope, "folder")
        if profile is None:
            return self._status_record(file, scope, ExtractionStatus.UNSUPPORTED.value)
        if known_change_keys is not None and known_change_keys.get(file.id) == (
            remote_fingerprint(file),
            processing_fingerprint(
                file,
                profile,
                extractor_version=self.extractor_version,
                chunker_version=self.chunker_version,
            ),
        ):
            return self._status_record(file, scope, UNCHANGED_STATUS)
        try:
            payload = (
                await self.client.export_file(file.id, profile.export_mime_type)
                if profile.export_mime_type
                else await self.client.download_file(file.id)
            )
            result = self.extractor(payload, file.mime_type, profile=profile, limits=self.extraction_limits)
        except Exception as error:
            info = classify_provider_error(error)
            tombstone = self.tombstone_for_error(file, error, scope=scope)
            if tombstone is not None:
                return tombstone
            if self.retry_work_store is not None and info.retryable:
                self.retry_work_store.schedule_failure(
                    scope_identity=self.scope_identity(scope) if scope else "unscoped",
                    source_id=file.id,
                    operation="materialize",
                    payload={"mime_type": file.mime_type},
                    error=error,
                )
            return self._status_record(file, scope, f"provider-{info.classification.value}", info.reason)
        if result.status is not ExtractionStatus.INDEXED or result.text is None:
            return self._status_record(file, scope, result.status.value, result.reason)
        return map_drive_file(
            file,
            workspace_id=self.workspace_id,
            body=result.text,
            extraction_status=result.status.value,
            extraction_reason=result.reason,
            scope_memberships=self._memberships_for(file.id, scope),
            clock=self.clock,
            extractor_version=self.extractor_version,
            chunker_version=self.chunker_version,
        )

    def tombstone_for_change(self, change: DriveChange, *, scope: DriveScope | None = None) -> Record | None:
        if not change.removed and not (change.file and change.file.trashed):
            return None
        file = change.file or DriveFile(change.file_id, "", "")
        remaining = self._discard_scope_membership(change.file_id, scope) if scope else ()
        if remaining:
            return self._status_record(
                file,
                None,
                "provider-definitive",
                "removed" if change.removed else "trashed",
                scope_memberships=remaining,
            )
        return map_drive_file(
            file,
            workspace_id=self.workspace_id,
            extraction_status="tombstone",
            extraction_reason="removed" if change.removed else "trashed",
            scope_memberships=(),
            clock=self.clock,
            status=RecordStatus.ARCHIVED,
            deleted=True,
            extractor_version=self.extractor_version,
            chunker_version=self.chunker_version,
        )

    def tombstone_for_error(
        self,
        file: DriveFile,
        error: BaseException,
        *,
        scope: DriveScope | None = None,
    ) -> Record | None:
        """Map only confirmed provider-side record loss to an archived record."""
        info = classify_provider_error(error)
        if not info.tombstone:
            return None
        remaining = self._discard_scope_membership(file.id, scope) if scope else ()
        if remaining:
            return self._status_record(
                file,
                None,
                f"provider-{info.classification.value}",
                info.reason,
                scope_memberships=remaining,
            )
        if info.reason:
            reason = info.reason
        elif info.status_code == 404:
            reason = "not-found"
        elif info.status_code == 410:
            reason = "gone"
        else:
            reason = "permission-lost"
        return map_drive_file(
            file,
            workspace_id=self.workspace_id,
            extraction_status="tombstone",
            extraction_reason=reason,
            scope_memberships=(),
            clock=self.clock,
            status=RecordStatus.ARCHIVED,
            deleted=True,
            extractor_version=self.extractor_version,
            chunker_version=self.chunker_version,
        )

    def _status_record(
        self,
        file: DriveFile,
        scope: DriveScope | None,
        status: str,
        reason: str | None = None,
        scope_memberships: tuple[str, ...] | None = None,
    ) -> Record:
        return map_drive_file(
            file,
            workspace_id=self.workspace_id,
            extraction_status=status,
            extraction_reason=reason,
            scope_memberships=scope_memberships or self._memberships_for(file.id, scope),
            clock=self.clock,
            extractor_version=self.extractor_version,
            chunker_version=self.chunker_version,
        )

    def _add_scope_membership(self, record: Record, scope: DriveScope | None) -> None:
        if scope is None:
            return
        memberships = self._add_scope_membership_for_source(record.source_id, scope)
        record.metadata["scope_memberships"] = list(memberships)

    def _add_scope_membership_for_source(
        self,
        source_id: str,
        scope: DriveScope | None,
    ) -> tuple[str, ...]:
        if scope is None:
            return self.membership_store.memberships_for(self.workspace_id, source_id)
        return self.membership_store.add(
            self.workspace_id,
            source_id,
            self.scope_identity(scope),
        )

    def _discard_scope_membership(
        self,
        source_id: str,
        scope: DriveScope | None,
    ) -> tuple[str, ...]:
        if scope is None:
            return self.membership_store.memberships_for(self.workspace_id, source_id)
        return self.membership_store.discard(
            self.workspace_id,
            source_id,
            self.scope_identity(scope),
        )

    def _memberships_for(self, source_id: str, scope: DriveScope | None) -> tuple[str, ...]:
        memberships = self.membership_store.memberships_for(self.workspace_id, source_id)
        if memberships or scope is None:
            return memberships
        return (self.scope_identity(scope),)

    def _record_visible_in_scope(self, record: Record, scope: DriveScope) -> bool:
        return self.scope_identity(scope) in record.metadata.get("scope_memberships", ())


__all__ = [
    "DETERMINISTIC_MATERIALIZATION_STATUSES",
    "DriveContentClient",
    "GoogleDriveContentSource",
    "UNCHANGED_STATUS",
]
