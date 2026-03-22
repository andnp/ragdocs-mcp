from click.testing import CliRunner

from src import cli as cli_module
def test_rebuild_index_cmd_delegates_to_shared_runner(monkeypatch) -> None:
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        cli_module,
        "run_rebuild_command",
        lambda **kwargs: captured.update(kwargs),
    )

    runner = CliRunner()
    result = runner.invoke(
        cli_module.cli,
        ["rebuild-index", "--project", "project-a", "--all-projects"],
    )

    assert result.exit_code == 0, result.output
    assert captured["project"] == "project-a"
    assert captured["all_projects"] is True
    assert captured["ensure_runtime_auto_registration"] is cli_module._ensure_runtime_auto_registration
    assert captured["emit"] is cli_module.click.echo


def test_rebuild_index_cmd_reports_runner_errors(monkeypatch) -> None:
    monkeypatch.setattr(
        cli_module,
        "run_rebuild_command",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("daemon rebuild failed")),
    )

    runner = CliRunner()
    result = runner.invoke(cli_module.cli, ["rebuild-index"])

    assert result.exit_code == 1
    assert "Error: daemon rebuild failed" in result.output
