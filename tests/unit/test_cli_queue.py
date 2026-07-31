from __future__ import annotations

import json

from click.testing import CliRunner

from ragdocs import cli as cli_module


def test_queue_purge_requires_yes() -> None:
    runner = CliRunner()

    result = runner.invoke(cli_module.cli, ["queue", "purge"])

    assert result.exit_code == 2
    assert "Refusing to purge queue state without --yes." in result.output


def test_queue_purge_requests_daemon_route_and_supports_json(
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        cli_module,
        "_request_daemon_json",
        lambda path, payload, *, project_override, auto_start, allow_error: captured.update(
            {
                "path": path,
                "payload": payload,
                "project_override": project_override,
                "auto_start": auto_start,
                "allow_error": allow_error,
            }
        )
        or {
            "status": "ok",
            "queue_db_path": "/runtime/queue.db",
            "purged_state": "scheduled",
            "purged_counts": {
                "pending": 0,
                "scheduled": 2,
                "failed": 0,
            },
            "pending_count": 1,
            "scheduled_count": 0,
            "running_count": 0,
            "failed_count": 1,
            "worker_running": True,
            "backpressure_limit": 5,
            "backpressure_utilization": 0.2,
            "task_counts": {},
            "recent_failures": [],
            "pending_tasks": [],
            "scheduled_tasks": [],
        },
    )

    runner = CliRunner()
    result = runner.invoke(
        cli_module.cli,
        [
            "queue",
            "purge",
            "--project",
            "project-a",
            "--state",
            "scheduled",
            "--yes",
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured == {
        "path": "/api/admin/tasks/purge",
        "payload": {
            "state": "scheduled",
            "confirm": True,
        },
        "project_override": "project-a",
        "auto_start": False,
        "allow_error": True,
    }
    assert json.loads(result.output) == {
        "status": "ok",
        "queue_db_path": "/runtime/queue.db",
        "purged_state": "scheduled",
        "purged_counts": {
            "pending": 0,
            "scheduled": 2,
            "failed": 0,
        },
        "pending_count": 1,
        "scheduled_count": 0,
        "running_count": 0,
        "failed_count": 1,
        "worker_running": True,
        "backpressure_limit": 5,
        "backpressure_utilization": 0.2,
        "task_counts": {},
        "recent_failures": [],
        "pending_tasks": [],
        "scheduled_tasks": [],
    }
