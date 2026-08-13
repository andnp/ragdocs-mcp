"""Narrow, bounded Google Drive v3 transport for the source adapter."""

import asyncio
import io
from collections.abc import Callable, Mapping
from typing import Any, Protocol, cast

from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build

from mcp_markdown_ragdocs.gdrive.models import (
    DriveChange,
    DriveChangePage,
    DriveFile,
    DriveFilePage,
    DriveScope,
    DriveStartPageToken,
    DriveWatchChannel,
)
from mcp_markdown_ragdocs.gdrive.gate import DriveRequestGate
from mcp_markdown_ragdocs.gdrive.session import DriveCredentialSession

FILE_FIELDS = (
    "nextPageToken,files("
    "id,name,mimeType,modifiedTime,size,md5Checksum,sha256Checksum,webViewLink,"
    "parents,driveId,trashed,shortcutDetails(targetId,targetMimeType))"
)
FILE_RESOURCE_FIELDS = (
    "id,name,mimeType,modifiedTime,size,md5Checksum,sha256Checksum,webViewLink,"
    "parents,driveId,trashed,shortcutDetails(targetId,targetMimeType)"
)
CHANGE_FIELDS = (
    "nextPageToken,newStartPageToken,changes("
    "fileId,removed,file("
    "id,name,mimeType,modifiedTime,size,md5Checksum,sha256Checksum,webViewLink,"
    "parents,driveId,trashed,shortcutDetails(targetId,targetMimeType)))"
)


class DriveRequest(Protocol):
    def execute(self) -> Any: ...


class DriveFilesResource(Protocol):
    def list(self, **kwargs: object) -> DriveRequest: ...

    def get(self, **kwargs: object) -> DriveRequest: ...

    def get_media(self, **kwargs: object) -> DriveRequest: ...

    def export(self, **kwargs: object) -> DriveRequest: ...


class DriveChangesResource(Protocol):
    def getStartPageToken(self, **kwargs: object) -> DriveRequest: ...

    def list(self, **kwargs: object) -> DriveRequest: ...

    def watch(self, **kwargs: object) -> DriveRequest: ...


class DriveChannelsResource(Protocol):
    def stop(self, **kwargs: object) -> DriveRequest: ...


class DriveService(Protocol):
    def files(self) -> DriveFilesResource: ...

    def changes(self) -> DriveChangesResource: ...

    def channels(self) -> DriveChannelsResource: ...


ServiceFactory = Callable[[Credentials], DriveService]


class GoogleDriveClient:
    """Injectable Drive transport with bounded list and media requests."""

    def __init__(
        self,
        session: DriveCredentialSession,
        *,
        service: DriveService | None = None,
        service_factory: ServiceFactory | None = None,
        max_page_size: int = 1000,
        max_download_bytes: int = 25 * 1024 * 1024,
        request_gate: DriveRequestGate | None = None,
    ) -> None:
        if max_page_size < 1:
            raise ValueError("max_page_size must be positive")
        if max_download_bytes < 1:
            raise ValueError("max_download_bytes must be positive")
        self._session = session
        self._service = service
        self._service_factory = service_factory or self._build_service
        self._max_page_size = max_page_size
        self._max_download_bytes = max_download_bytes
        self._request_gate = request_gate

    @staticmethod
    def _build_service(credentials: Credentials) -> DriveService:
        return cast(DriveService, build("drive", "v3", credentials=credentials))

    def _get_service(self) -> DriveService:
        if self._service is None:
            self._service = self._service_factory(self._session.get_credentials())
        return self._service

    def _page_size(self, requested: int) -> int:
        if requested < 1:
            raise ValueError("page_size must be positive")
        return min(requested, self._max_page_size)

    def _media_bytes(self, result: Any) -> bytes:
        if isinstance(result, bytes):
            content = result
        elif isinstance(result, (bytearray, memoryview)):
            content = bytes(result)
        elif isinstance(result, io.BufferedIOBase):
            content = result.read(self._max_download_bytes + 1)
        elif isinstance(result, Mapping):
            content = self._media_bytes_from_value(result.get("body", result.get("content", result.get("data"))))
        else:
            content = self._media_bytes_from_value(
                next(
                    (getattr(result, name) for name in ("content", "body", "data") if hasattr(result, name)),
                    None,
                )
            )
        if len(content) > self._max_download_bytes:
            raise ValueError("Google Drive media response exceeded configured byte limit")
        return content

    def _media_bytes_from_value(self, value: Any) -> bytes:
        if value is None:
            raise TypeError("Drive media response did not contain bytes")
        return self._media_bytes(value)

    async def list_files_page(
        self,
        scope: DriveScope,
        *,
        page_token: str | None = None,
        page_size: int = 1000,
    ) -> DriveFilePage:
        kwargs: dict[str, object] = {
            "includeItemsFromAllDrives": True,
            "supportsAllDrives": True,
            "pageSize": self._page_size(page_size),
            "fields": FILE_FIELDS,
            "orderBy": "modifiedTime, name",
            "q": "trashed = false",
        }
        if scope.shared_drive_id:
            kwargs.update(corpora="drive", driveId=scope.shared_drive_id)
        elif scope.include_shared_with_me:
            kwargs["q"] = "trashed = false and sharedWithMe = true"
        if page_token:
            kwargs["pageToken"] = page_token
        result = await self._execute(lambda: self._get_service().files().list(**kwargs))
        return DriveFilePage(
            files=tuple(DriveFile.from_api(item) for item in result.get("files", ())),
            next_page_token=str(result.get("nextPageToken") or "") or None,
        )

    async def get_start_page_token(self, scope: DriveScope) -> DriveStartPageToken:
        kwargs: dict[str, object] = {"supportsAllDrives": True}
        if scope.shared_drive_id:
            kwargs["driveId"] = scope.shared_drive_id
        result = await self._execute(
            lambda: self._get_service().changes().getStartPageToken(**kwargs)
        )
        return DriveStartPageToken(str(result.get("startPageToken") or ""))

    async def list_changes_page(
        self,
        scope: DriveScope,
        page_token: str,
        *,
        page_size: int = 1000,
    ) -> DriveChangePage:
        kwargs: dict[str, object] = {
            "includeItemsFromAllDrives": True,
            "supportsAllDrives": True,
            "pageToken": page_token,
            "pageSize": self._page_size(page_size),
            "includeRemoved": True,
            "fields": CHANGE_FIELDS,
        }
        if scope.shared_drive_id:
            kwargs["driveId"] = scope.shared_drive_id
        result = await self._execute(lambda: self._get_service().changes().list(**kwargs))
        return DriveChangePage(
            changes=tuple(DriveChange.from_api(item) for item in result.get("changes", ())),
            next_page_token=str(result.get("nextPageToken") or "") or None,
            new_start_page_token=str(result.get("newStartPageToken") or "") or None,
        )

    async def get_file_metadata(self, file_id: str) -> DriveFile:
        result = await self._execute(
            lambda: self._get_service().files().get(
                fileId=file_id,
                fields=FILE_RESOURCE_FIELDS,
                supportsAllDrives=True,
            )
        )
        return DriveFile.from_api(result)

    async def download_file(self, file_id: str) -> bytes:
        result = await self._execute(
            lambda: self._get_service().files().get_media(fileId=file_id)
        )
        return self._media_bytes(result)

    async def export_file(self, file_id: str, export_mime_type: str) -> bytes:
        result = await self._execute(
            lambda: self._get_service().files().export(
                fileId=file_id,
                mimeType=export_mime_type,
            )
        )
        return self._media_bytes(result)

    async def watch_changes(
        self,
        scope: DriveScope,
        page_token: str,
        *,
        channel_id: str,
        address: str,
        token: str | None = None,
    ) -> DriveWatchChannel:
        body: dict[str, str] = {"id": channel_id, "type": "web_hook", "address": address}
        if token:
            body["token"] = token
        kwargs: dict[str, object] = {"pageToken": page_token, "body": body}
        if scope.shared_drive_id:
            kwargs["driveId"] = scope.shared_drive_id
        result = await self._execute(lambda: self._get_service().changes().watch(**kwargs))
        return DriveWatchChannel(
            channel_id=str(result.get("id") or channel_id),
            resource_id=str(result.get("resourceId") or "") or None,
            expiration=int(str(result.get("expiration") or "0")),
            address=address,
        )

    async def stop_channel(self, channel_id: str, resource_id: str | None = None) -> None:
        body = {"id": channel_id}
        if resource_id:
            body["resourceId"] = resource_id
        await self._execute(lambda: self._get_service().channels().stop(body=body))

    async def _execute(self, request_factory: Callable[[], DriveRequest]) -> Any:
        """Execute one provider request in a worker thread without unbounded retry."""
        def execute_request() -> Any:
            request = request_factory()
            return request.execute()

        if self._request_gate is not None:
            return await asyncio.to_thread(self._request_gate.run, execute_request)
        return await asyncio.to_thread(execute_request)


__all__ = ["GoogleDriveClient"]
