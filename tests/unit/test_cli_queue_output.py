from __future__ import annotations

from mcp_markdown_ragdocs.cli_utils.queue_output import (
    _emit_git_refresh_progress,
    _emit_queue_task_section,
    _filter_queue_status_payload,
)


def test_filter_queue_status_payload_selects_details_and_limits_items() -> None:
    payload: dict[str, object] = {
        "pending_tasks": [{"task_id": "p1"}, {"task_id": "p2"}],
        "scheduled_tasks": [{"task_id": "s1"}],
        "recent_failures": [{"task_id": "f1"}, {"task_id": "f2"}],
    }

    filtered = _filter_queue_status_payload(
        payload,
        state="failed",
        limit=1,
    )

    assert filtered["pending_tasks"] == []
    assert filtered["scheduled_tasks"] == []
    assert filtered["recent_failures"] == [{"task_id": "f1"}]
    assert filtered["detail_state_filter"] == "failed"
    assert filtered["detail_limit"] == 1


def test_queue_output_renders_task_details_and_git_progress(capsys) -> None:
    _emit_queue_task_section(
        "Pending:",
        [
            {
                "task_name": "index_document",
                "task_id": "task-1",
                "state": "pending",
                "source_queue": "pending",
                "priority": 3,
                "eta": "later",
                "retries": "2",
            }
        ],
        details=True,
    )
    _emit_git_refresh_progress(
        [
            {
                "repository_path": "/repo",
                "state": "running",
                "processed_count": 2,
                "discovered_count": 5,
            },
            {"repository_path": "/other", "state": "queued"},
        ]
    )

    output = capsys.readouterr().out
    assert "Pending:" in output
    assert "index_document" in output
    assert "    retries: 2" in output
    assert "Git refresh:" in output
    assert "  /repo: running (2/5)" in output
    assert "  /other: queued" in output
