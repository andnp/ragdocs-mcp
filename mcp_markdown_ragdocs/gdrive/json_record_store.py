"""Shared schema-versioned JSON envelope for gdrive persistent stores."""

from __future__ import annotations

import json
from pathlib import Path
from typing import overload

from searchkernel.api import atomic_write_json


@overload
def read_json_envelope(
    path: Path, *, schema_version: int, key: str, expected_type: type[dict[str, object]]
) -> dict[str, object] | None: ...
@overload
def read_json_envelope(
    path: Path, *, schema_version: int, key: str, expected_type: type[list[object]]
) -> list[object] | None: ...
def read_json_envelope(
    path: Path, *, schema_version: int, key: str, expected_type: type[object]
) -> object | None:
    """Load one schema-versioned JSON envelope, or None if missing or invalid."""

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return None
    if not isinstance(raw, dict) or raw.get("schema_version") != schema_version:
        return None
    value = raw.get(key)
    if not isinstance(value, expected_type):
        return None
    return value


def write_json_envelope(
    path: Path, *, schema_version: int, key: str, value: dict[str, object] | list[object]
) -> None:
    """Atomically persist one schema-versioned JSON envelope."""

    atomic_write_json(path, {"schema_version": schema_version, key: value})


__all__ = ["read_json_envelope", "write_json_envelope"]
