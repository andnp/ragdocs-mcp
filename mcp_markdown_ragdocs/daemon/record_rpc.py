"""Wire-boundary serialization helpers for daemon Record ingestion."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from searchkernel.api import Record


class RecordSerializationError(ValueError):
    """Raised when a daemon payload is not a serialized Record."""


def serialize_record(record: Record) -> dict[str, Any]:
    """Serialize a Record using the domain model's ISO datetime format."""
    return record.to_dict()


def deserialize_record(data: object) -> Record:
    """Deserialize one JSON-compatible Record dictionary.

    Datetimes are required to be ISO 8601 strings at the RPC boundary. The
    domain model performs the actual conversion so this wire format stays in
    sync with other Record producers and consumers.
    """
    if not isinstance(data, dict):
        raise RecordSerializationError("record must be an object")
    if not all(isinstance(key, str) for key in data):
        raise RecordSerializationError("record keys must be strings")

    for field_name in ("created_at", "updated_at"):
        value = data.get(field_name)
        if not isinstance(value, str):
            raise RecordSerializationError(
                f"{field_name} must be an ISO datetime string"
            )
        try:
            datetime.fromisoformat(value)
        except ValueError as exc:
            raise RecordSerializationError(
                f"{field_name} must be an ISO datetime string"
            ) from exc

    try:
        return Record.from_dict(data)
    except KeyError as exc:
        raise RecordSerializationError(f"missing required field: {exc.args[0]}") from exc
    except (TypeError, ValueError) as exc:
        raise RecordSerializationError(str(exc)) from exc
