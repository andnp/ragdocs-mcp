import pytest
from click.testing import CliRunner
from src.cli import cli


@pytest.fixture
def multi_project_workspace(tmp_path, monkeypatch):
    project_a = tmp_path / "project_a"
    project_a.mkdir()
    (project_a / "docs").mkdir()
    (project_a / "docs" / "readme.md").write_text("# Project A")

    project_b = tmp_path / "project_b"
    project_b.mkdir()
    (project_b / "docs").mkdir()
    (project_b / "docs" / "readme.md").write_text("# Project B")

    monkeypatch.setenv("HOME", str(tmp_path))

    config_dir = tmp_path / ".config" / "mcp-markdown-ragdocs"
    config_dir.mkdir(parents=True)
    (config_dir / "config.toml").write_text(f"""
[[projects]]
name = "project_a"
path = "{project_a}"

[[projects]]
name = "project_b"
path = "{project_b}"

[indexing]
documents_path = "./docs"
""")

    return {"project_a": project_a, "project_b": project_b, "tmp": tmp_path}


def test_cli_check_config_shows_projects(multi_project_workspace, monkeypatch):
    monkeypatch.chdir(multi_project_workspace["project_a"])

    runner = CliRunner()
    result = runner.invoke(cli, ["check-config"])

    assert result.exit_code == 0
    assert "project_a" in result.output
    assert "project_b" in result.output
    assert "Active Project" in result.output


def test_cli_project_detection_from_subdirectory(multi_project_workspace, monkeypatch):
    monkeypatch.chdir(multi_project_workspace["project_a"] / "docs")

    runner = CliRunner()
    result = runner.invoke(cli, ["check-config"])

    assert result.exit_code == 0
    assert "project_a" in result.output
    assert "Active Project" in result.output
