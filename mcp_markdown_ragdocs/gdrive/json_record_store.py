"""Shared schema-versioned JSON envelope for gdrive persistent stores."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import overload

from searchkernel.api import atomic_write_json


@dataclass(frozen=True, slots=True)
class JsonEnvelopeStore:
    """Atomic JSON file shaped as {"schema_version": V, <key>: value}."""

    path: Path
    schema_version: int
    key: str

    @overload
    def read(self, expected_type: type[dict[str, object]]) -> dict[str, object] | None: ...
    @overload
    def read(self, expected_type: type[list[object]]) -> list[object] | None: ...
    def read(self, expected_type: type[object]) -> object | None:
        """Load the envelope's value, or None if missing, stale, or invalid."""

        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (FileNotFoundError, OSError, json.JSONDecodeError):
            return None
        if not isinstance(raw, dict) or raw.get("schema_version") != self.schema_version:
            return None
        value = raw.get(self.key)
        return value if isinstance(value, expected_type) else None

    def write(self, value: Mapping[str, object] | Sequence[object]) -> None:
        """Atomically persist the envelope's value."""

        atomic_write_json(self.path, {"schema_version": self.schema_version, self.key: value})


__all__ = ["JsonEnvelopeStore"]
