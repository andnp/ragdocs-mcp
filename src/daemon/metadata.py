from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import json
from pathlib import Path
from typing import Any, cast


@dataclass(frozen=True)
class DaemonMetadata:
    pid: int
    started_at: float
    status: str
    daemon_scope: str = "global"
    transport: str = "zmq"
    socket_path: str | None = None
    binary_path: str | None = None
    version: str | None = None
    index_db_path: str | None = None
    queue_db_path: str | None = None

    @property
    def transport_endpoint(self) -> str | None:
        if self.socket_path:
            return f"ipc://{self.socket_path}"
        return None


def write_daemon_metadata(metadata_path: Path, metadata: DaemonMetadata) -> None:
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(
        json.dumps(asdict(metadata), sort_keys=True),
        encoding="utf-8",
    )


def read_daemon_metadata(metadata_path: Path) -> DaemonMetadata | None:
    if not metadata_path.exists():
        return None

    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None

    try:
        return DaemonMetadata(**cast(Any, _normalize_payload(payload)))
    except TypeError:
        return None


def remove_daemon_metadata(metadata_path: Path) -> None:
    try:
        metadata_path.unlink()
    except FileNotFoundError:
        return


def _normalize_payload(payload: dict[str, object]) -> dict[str, object]:
    allowed_fields = {field.name for field in fields(DaemonMetadata)}
    normalized = {key: value for key, value in payload.items() if key in allowed_fields}
    normalized.setdefault("daemon_scope", "global")
    normalized.setdefault("transport", "zmq")
    normalized.setdefault("socket_path", None)
    normalized.setdefault("binary_path", None)
    normalized.setdefault("version", None)
    normalized.setdefault("index_db_path", None)
    normalized.setdefault("queue_db_path", None)
    return normalized