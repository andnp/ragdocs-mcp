"""Read-only diagnostics for SQLite storage and its backing filesystem."""

from __future__ import annotations

import shutil
import sqlite3
from pathlib import Path
from urllib.parse import quote


_PRAGMAS = (
    "journal_mode",
    "page_size",
    "page_count",
    "freelist_count",
    "max_page_count",
    "wal_autocheckpoint",
    "temp_store",
)


def _sidecar_size(path: Path) -> int:
    try:
        return path.stat().st_size
    except FileNotFoundError:
        return 0
    except OSError:
        return -1


def _read_pragma(connection: sqlite3.Connection, name: str) -> object:
    row = connection.execute(f"PRAGMA {name}").fetchone()
    return None if row is None else row[0]


def sqlite_storage_diagnostics(db_path: Path) -> dict[str, object]:
    """Return SQLite capacity details without opening the database for writes."""

    db_path = db_path.resolve()
    payload: dict[str, object] = {
        "path": str(db_path),
        "exists": db_path.exists(),
        "db_size_bytes": _sidecar_size(db_path),
        "wal_size_bytes": _sidecar_size(Path(f"{db_path}-wal")),
        "shm_size_bytes": _sidecar_size(Path(f"{db_path}-shm")),
    }
    try:
        usage = shutil.disk_usage(db_path.parent)
    except OSError as error:
        payload["filesystem_error"] = str(error)
    else:
        payload["filesystem"] = {
            "total_bytes": usage.total,
            "used_bytes": usage.used,
            "free_bytes": usage.free,
        }

    if not db_path.exists():
        payload["status"] = "missing"
        return payload

    uri = f"file:{quote(str(db_path), safe='/')}?mode=ro"
    try:
        with sqlite3.connect(uri, uri=True, timeout=1.0) as connection:
            payload["sqlite"] = {
                name: _read_pragma(connection, name) for name in _PRAGMAS
            }
    except (OSError, sqlite3.Error) as error:
        payload["status"] = "error"
        payload["error"] = str(error)
    else:
        payload["status"] = "ready"
    return payload
