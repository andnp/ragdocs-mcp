from __future__ import annotations

from click.testing import CliRunner

from mcp_markdown_ragdocs import cli as cli_module


def test_prune_old_git_diffs_requests_daemon_route_dry_run(monkeypatch) -> None:
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
            "would_delete": 3,
            "workspace_id": "project-a",
            "max_age_days": 30,
        },
    )

    runner = CliRunner()
    result = runner.invoke(
        cli_module.cli,
        [
            "records",
            "prune-old-git-diffs",
            "--project",
            "project-a",
            "--workspace",
            "project-a",
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured == {
        "path": "/api/admin/records/prune-old-git-diffs",
        "payload": {"workspace_id": "project-a", "confirm": False},
        "project_override": "project-a",
        "auto_start": False,
        "allow_error": True,
    }
    assert "Would delete 3 old diff chunk(s)" in result.output
    assert "Pass --yes to delete." in result.output


def test_prune_old_git_diffs_yes_deletes_and_reports_count(monkeypatch) -> None:
    monkeypatch.setattr(
        cli_module,
        "_request_daemon_json",
        lambda path, payload, *, project_override, auto_start, allow_error: {
            "status": "ok",
            "deleted": 5,
            "workspace_id": "project-a",
            "max_age_days": 30,
        },
    )

    runner = CliRunner()
    result = runner.invoke(
        cli_module.cli,
        ["records", "prune-old-git-diffs", "--workspace", "project-a", "--yes"],
    )

    assert result.exit_code == 0, result.output
    assert "Deleted 5 old diff chunk(s)." in result.output


def test_prune_old_git_diffs_supports_json_output(monkeypatch) -> None:
    monkeypatch.setattr(
        cli_module,
        "_request_daemon_json",
        lambda path, payload, *, project_override, auto_start, allow_error: {
            "status": "ok",
            "would_delete": 0,
            "workspace_id": "project-a",
            "max_age_days": 30,
        },
    )

    runner = CliRunner()
    result = runner.invoke(
        cli_module.cli,
        ["records", "prune-old-git-diffs", "--workspace", "project-a", "--json"],
    )

    assert result.exit_code == 0, result.output
    assert '"would_delete": 0' in result.output


def test_prune_old_git_diffs_requires_workspace_option() -> None:
    runner = CliRunner()

    result = runner.invoke(cli_module.cli, ["records", "prune-old-git-diffs"])

    assert result.exit_code == 2
    assert "--workspace" in result.output
