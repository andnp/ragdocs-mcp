"""Tests for Google Drive source health states."""

from pathlib import Path

import pytest

from mcp_markdown_ragdocs.gdrive.health import (
    DriveHealthStatus,
    DriveScopeHealth,
    DriveSourceHealth,
    GDriveHealthStore,
)


@pytest.mark.parametrize(
    ("scope", "available", "expected"),
    [
        (DriveScopeHealth("shared-with-me"), True, DriveHealthStatus.EMPTY),
        (
            DriveScopeHealth(
                "shared-with-me",
                indexed_records=2,
                remote_records=3,
                last_success_at=80,
            ),
            True,
            DriveHealthStatus.STALE,
        ),
        (
            DriveScopeHealth(
                "shared-with-me",
                indexed_records=2,
                remote_records=3,
                acl_complete=False,
                last_success_at=99,
            ),
            True,
            DriveHealthStatus.ACL_INCOMPLETE,
        ),
        (
            DriveScopeHealth(
                "shared-with-me",
                indexed_records=2,
                remote_records=3,
                last_success_at=99,
            ),
            False,
            DriveHealthStatus.UNAVAILABLE,
        ),
        (
            DriveScopeHealth(
                "shared-with-me",
                indexed_records=2,
                remote_records=3,
                last_success_at=99,
            ),
            True,
            DriveHealthStatus.HEALTHY,
        ),
    ],
)
def test_drive_health_distinguishes_source_states(
    scope: DriveScopeHealth,
    available: bool,
    expected: DriveHealthStatus,
) -> None:
    """
    Classify empty, stale, ACL-incomplete, unavailable, and healthy inputs.
    """
    health = DriveSourceHealth.evaluate(
        "workspace",
        (scope,),
        available=available,
        observed_at=100,
        stale_after_seconds=10,
    )

    assert health.status is expected
    assert health.source_kind == "google_drive"
    assert health.workspace_id == "workspace"


def test_drive_health_retains_scope_specific_data(tmp_path: Path) -> None:
    """
    Persist counts, ACL gaps, freshness, watch mode, and the last error.
    """
    health = DriveSourceHealth.evaluate(
        "workspace",
        (
            DriveScopeHealth(
                "shared-with-me",
                indexed_records=4,
                remote_records=5,
                last_success_at=99,
            ),
            DriveScopeHealth(
                "shared-drive:drive-1",
                indexed_records=1,
                remote_records=2,
                acl_complete=False,
                last_success_at=98,
                last_error="permission denied",
            ),
        ),
        observed_at=100,
        stale_after_seconds=10,
        watch_mode="poll",
        last_error="renewal failed",
    )
    store = GDriveHealthStore(tmp_path)

    store.save(health)
    payload = store.load("workspace")

    assert payload is not None
    assert payload["status"] == "acl-incomplete"
    source = payload["source"]
    assert isinstance(source, dict)
    assert source["indexed_records"] == 5
    assert source["remote_records"] == 7
    assert source["acl_incomplete_scopes"] == ["shared-drive:drive-1"]
    assert source["watch_mode"] == "poll"
    assert source["last_error"] == "renewal failed"
    assert source["scopes"][1]["last_error"] == "permission denied"
