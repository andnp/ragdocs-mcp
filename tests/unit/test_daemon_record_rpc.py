from datetime import UTC, datetime

import pytest
from searchkernel.domain import Record, RecordStatus

from mcp_markdown_ragdocs.daemon.record_rpc import (
    RecordSerializationError,
    deserialize_record,
    serialize_record,
)


def _record() -> Record:
    return Record(
        source_kind="gmail",
        source_id="gmail:message-1",
        title="Hello",
        body="Message body",
        created_at=datetime(2026, 7, 29, 12, 30, 1, tzinfo=UTC),
        updated_at=datetime(2026, 7, 29, 12, 31, 2, tzinfo=UTC),
        status=RecordStatus.ACTIVE,
        metadata={"labels": ["inbox"]},
        uri="https://example.test/message-1",
    )


def test_serialize_record_uses_iso_datetime_strings() -> None:
    payload = serialize_record(_record())

    assert payload["created_at"] == "2026-07-29T12:30:01+00:00"
    assert payload["updated_at"] == "2026-07-29T12:31:02+00:00"
    assert payload["status"] == "active"


def test_deserialize_record_round_trips_iso_datetimes() -> None:
    original = _record()

    restored = deserialize_record(serialize_record(original))

    assert restored == original


def test_deserialize_record_accepts_utc_z_suffix() -> None:
    payload = serialize_record(_record())
    payload["created_at"] = "2026-07-29T12:30:01Z"
    payload["updated_at"] = "2026-07-29T12:31:02Z"

    restored = deserialize_record(payload)

    assert restored.created_at == datetime(2026, 7, 29, 12, 30, 1, tzinfo=UTC)
    assert restored.updated_at == datetime(2026, 7, 29, 12, 31, 2, tzinfo=UTC)


@pytest.mark.parametrize(
    "payload, message",
    [
        ([], "record must be an object"),
        ({"source_id": "note:1"}, "created_at must be an ISO datetime string"),
        (
            {"created_at": "2026-07-29T12:30:01Z", "updated_at": "invalid"},
            "updated_at must be an ISO datetime string",
        ),
    ],
)
def test_deserialize_record_rejects_invalid_wire_data(payload, message: str) -> None:
    with pytest.raises(RecordSerializationError, match=message):
        deserialize_record(payload)
