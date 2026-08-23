from __future__ import annotations

import pytest

from mcp_markdown_ragdocs.daemon.status_snapshot import StatusSnapshot


def test_status_snapshot_reuses_value_until_explicit_refresh() -> None:
    """Repeated reads reuse one value and expose age after the clock advances."""
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
    now = 17.0
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
    assert second_status.stale is True
    assert second_status.age_seconds == 7.0


def test_status_snapshot_reports_failed_refresh_without_losing_last_value() -> None:
    """A failed refresh keeps the last value while exposing an error and staleness."""
    calls = 0

    def build() -> dict[str, object]:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("index unavailable")
        return {"value": calls}

    snapshot = StatusSnapshot(clock=lambda: 1.0)
    snapshot.read(build)
    value, status = snapshot.refresh(build)

    assert value == {"value": 1}
    assert status.stale is True
    assert status.error == "RuntimeError: index unavailable"


def test_status_snapshot_rejects_non_positive_stale_window() -> None:
    """A snapshot requires a positive freshness window for meaningful staleness."""
    with pytest.raises(ValueError, match="stale_after_seconds must be positive"):
        StatusSnapshot(stale_after_seconds=0.0)
