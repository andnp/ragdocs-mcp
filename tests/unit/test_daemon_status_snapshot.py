from __future__ import annotations

import pytest

from mcp_markdown_ragdocs.daemon.status_snapshot import StatusSnapshot


def test_status_snapshot_reuses_fresh_value() -> None:
    """Repeated reads reuse a fresh value without rebuilding it."""
    now = 10.0
    calls = 0

    def clock() -> float:
        return now

    def build() -> dict[str, object]:
        nonlocal calls
        calls += 1
        return {"value": calls}

    snapshot = StatusSnapshot(stale_after_seconds=5.0, clock=clock)
    first, first_status = snapshot.read(build)
    now = 14.0
    second, second_status = snapshot.read(build)

    assert first == {"value": 1}
    assert second == first
    assert calls == 1
    assert first_status.to_dict() == {
        "age_seconds": 0.0,
        "stale": False,
        "error": None,
        "stale_after_seconds": 5.0,
    }
    assert second_status.stale is False
    assert second_status.age_seconds == 4.0


def test_status_snapshot_refreshes_expired_value_on_read() -> None:
    """An expired read builds a new value before returning its status."""
    now = 10.0
    calls = 0

    def clock() -> float:
        return now

    def build() -> dict[str, object]:
        nonlocal calls
        calls += 1
        return {"value": calls}

    snapshot = StatusSnapshot(stale_after_seconds=5.0, clock=clock)
    snapshot.read(build)
    now = 16.0

    value, status = snapshot.read(build)

    assert value == {"value": 2}
    assert status.stale is False
    assert status.age_seconds == 0.0
    assert calls == 2


def test_status_snapshot_preserves_explicit_refresh() -> None:
    """An explicit refresh rebuilds a snapshot before it expires."""
    calls = 0

    def build() -> dict[str, object]:
        nonlocal calls
        calls += 1
        return {"value": calls}

    snapshot = StatusSnapshot(clock=lambda: 1.0)
    snapshot.read(build)
    value, status = snapshot.refresh(build)

    assert value == {"value": 2}
    assert status.stale is False
    assert calls == 2


def test_status_snapshot_reports_failed_refresh_without_losing_last_value() -> None:
    """A failed lazy refresh keeps the last value while exposing an error."""
    now = 1.0
    calls = 0

    def clock() -> float:
        return now

    def build() -> dict[str, object]:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("index unavailable")
        return {"value": calls}

    snapshot = StatusSnapshot(
        stale_after_seconds=5.0,
        retry_after_seconds=3.0,
        clock=clock,
    )
    snapshot.read(build)
    now = 7.0
    value, status = snapshot.read(build)

    assert value == {"value": 1}
    assert status.stale is True
    assert status.error == "RuntimeError: index unavailable"

    now = 9.0
    snapshot.read(build)
    assert calls == 2

    now = 10.0
    snapshot.read(build)
    assert calls == 3


def test_status_snapshot_rejects_non_positive_retry_window() -> None:
    """A snapshot requires a positive retry window for failed refreshes."""
    with pytest.raises(ValueError, match="retry_after_seconds must be positive"):
        StatusSnapshot(retry_after_seconds=0.0)


def test_status_snapshot_rejects_non_positive_stale_window() -> None:
    """A snapshot requires a positive freshness window for meaningful staleness."""
    with pytest.raises(ValueError, match="stale_after_seconds must be positive"):
        StatusSnapshot(stale_after_seconds=0.0)
