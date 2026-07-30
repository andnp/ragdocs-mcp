from datetime import UTC, datetime

from searchkernel.domain.models import Record
from searchkernel.domain.timestamp_policy import (
    DEFAULT_TIMESTAMP_SELECTOR,
    TimestampSelector,
)

_CREATED_AT = datetime(2020, 1, 1, tzinfo=UTC)
_UPDATED_AT = datetime(2026, 1, 1, tzinfo=UTC)


def _record() -> Record:
    return Record(
        source_kind="note",
        source_id="note:1",
        title="Title",
        body="Body",
        created_at=_CREATED_AT,
        updated_at=_UPDATED_AT,
    )


def test_default_selector_prefers_updated_at_over_created_at():
    assert DEFAULT_TIMESTAMP_SELECTOR.select(_record()) == _UPDATED_AT


def test_selector_can_be_configured_to_prefer_created_at():
    selector = TimestampSelector(prefer_updated_at=False)
    assert selector.select(_record()) == _CREATED_AT
