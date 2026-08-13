"""Typed values returned by the Google Drive provider boundary."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class DriveFile:
    id: str
    name: str
    mime_type: str
    modified_time: str | None = None
    size: int | None = None
    md5_checksum: str | None = None
    sha256_checksum: str | None = None
    web_view_link: str | None = None
    parents: tuple[str, ...] = ()
    drive_id: str | None = None
    trashed: bool = False
    shortcut_target_id: str | None = None
    shortcut_target_mime_type: str | None = None

    @classmethod
    def from_api(cls, item: Any) -> "DriveFile":
        if not isinstance(item, Mapping):
            return cls(id="", name="", mime_type="")
        shortcut = item.get("shortcutDetails")
        shortcut = shortcut if isinstance(shortcut, Mapping) else {}
        raw_size = item.get("size")
        try:
            size = int(raw_size) if raw_size is not None else None
        except (TypeError, ValueError):
            size = None
        return cls(
            id=str(item.get("id") or ""),
            name=str(item.get("name") or ""),
            mime_type=str(item.get("mimeType") or ""),
            modified_time=str(item["modifiedTime"]) if item.get("modifiedTime") else None,
            size=size,
            md5_checksum=str(item["md5Checksum"]) if item.get("md5Checksum") else None,
            sha256_checksum=str(item["sha256Checksum"]) if item.get("sha256Checksum") else None,
            web_view_link=str(item["webViewLink"]) if item.get("webViewLink") else None,
            parents=tuple(str(parent) for parent in item.get("parents", ()) if parent),
            drive_id=str(item["driveId"]) if item.get("driveId") else None,
            trashed=bool(item.get("trashed", False)),
            shortcut_target_id=str(shortcut["targetId"]) if shortcut.get("targetId") else None,
            shortcut_target_mime_type=(
                str(shortcut["targetMimeType"])
                if shortcut.get("targetMimeType")
                else None
            ),
        )


@dataclass(frozen=True, slots=True)
class DriveChange:
    file_id: str
    removed: bool
    file: DriveFile | None = None

    @classmethod
    def from_api(cls, item: Any) -> "DriveChange":
        if not isinstance(item, Mapping):
            return cls(file_id="", removed=True)
        file = item.get("file")
        return cls(
            file_id=str(
                item.get("fileId")
                or (file.get("id") if isinstance(file, Mapping) else "")
                or ""
            ),
            removed=bool(item.get("removed", False)),
            file=DriveFile.from_api(file) if isinstance(file, Mapping) else None,
        )


@dataclass(frozen=True, slots=True)
class DriveScope:
    workspace_id: str
    shared_drive_id: str | None = None
    include_shared_with_me: bool = False

    @property
    def is_shared_drive(self) -> bool:
        return self.shared_drive_id is not None


@dataclass(frozen=True, slots=True)
class DriveWorkspace:
    workspace_id: str
    scopes: tuple[DriveScope, ...] = ()


@dataclass(frozen=True, slots=True)
class DriveWatchChannel:
    channel_id: str
    resource_id: str | None
    expiration: int
    address: str
    status: str = "active"
    last_error: str | None = None


__all__ = [
    "DriveChange",
    "DriveFile",
    "DriveScope",
    "DriveWatchChannel",
    "DriveWorkspace",
]
