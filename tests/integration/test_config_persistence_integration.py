import os
import tomllib
from pathlib import Path

import pytest
from click.testing import CliRunner

from src.cli import cli
from src.config import detect_project


@pytest.fixture
def temp_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    return tmp_path


def test_e2e_arbitrary_path_remains_transient(temp_home):
    """
    End-to-end test: arbitrary path override does not mutate global config.
    """
    config_dir = temp_home / ".config" / "mcp-markdown-ragdocs"
    config_path = config_dir / "config.toml"

    project_dir = temp_home / "my-awesome-project"
    project_dir.mkdir(parents=True)

    result = detect_project(
        cwd=Path("/somewhere/else"), projects=None, project_override=str(project_dir)
    )

    assert result == "my-awesome-project"
    assert not config_path.exists()

    result_again = detect_project(
        cwd=Path("/somewhere/else"),
        projects=None,
        project_override="my-awesome-project",
    )

    assert result_again is None


def test_e2e_multiple_arbitrary_paths_with_conflicts(temp_home):
    """
    End-to-end test: multiple arbitrary paths with same dir name get stable transient names.
    """
    project_dir_1 = temp_home / "projects" / "frontend"
    project_dir_1.mkdir(parents=True)

    project_dir_2 = temp_home / "work" / "frontend"
    project_dir_2.mkdir(parents=True)

    project_dir_3 = temp_home / "personal" / "frontend"
    project_dir_3.mkdir(parents=True)

    result_1 = detect_project(
        cwd=Path("/somewhere"), projects=None, project_override=str(project_dir_1)
    )
    assert result_1 == "frontend"

    result_2 = detect_project(
        cwd=Path("/somewhere"), projects=None, project_override=str(project_dir_2)
    )
    assert result_2 == "frontend"

    result_3 = detect_project(
        cwd=Path("/somewhere"), projects=None, project_override=str(project_dir_3)
    )
    assert result_3 == "frontend"

    config_path = temp_home / ".config" / "mcp-markdown-ragdocs" / "config.toml"
    assert not config_path.exists()


def test_e2e_explicit_registration_available_in_next_detect(temp_home):
    """
    End-to-end test: explicitly registered projects remain available for later detection.
    """
    config_dir = temp_home / ".config" / "mcp-markdown-ragdocs"
    config_dir.mkdir(parents=True)
    config_path = config_dir / "config.toml"

    project_dir = temp_home / "test-project"
    project_dir.mkdir(parents=True)

    subdirectory = project_dir / "src" / "lib"
    subdirectory.mkdir(parents=True)

    config_path.write_text(
        f"""
[[projects]]
name = "test-project"
path = "{project_dir}"
"""
    )

    result_by_cwd = detect_project(
        cwd=subdirectory, projects=None, project_override=None
    )
    assert result_by_cwd == "test-project"

    result_by_name = detect_project(
        cwd=Path("/other/location"), projects=None, project_override="test-project"
    )
    assert result_by_name == "test-project"


def test_e2e_config_preserved_across_transient_overrides(temp_home):
    """
    End-to-end test: transient overrides leave unrelated config untouched.
    """
    config_dir = temp_home / ".config" / "mcp-markdown-ragdocs"
    config_dir.mkdir(parents=True)
    config_path = config_dir / "config.toml"

    config_path.write_text(
        """
[server]
host = "0.0.0.0"
port = 9000

[search]
semantic_weight = 2.0
keyword_weight = 0.5
"""
    )

    project_dir_1 = temp_home / "project-1"
    project_dir_1.mkdir()
    detect_project(
        cwd=Path("/elsewhere"), projects=None, project_override=str(project_dir_1)
    )

    project_dir_2 = temp_home / "project-2"
    project_dir_2.mkdir()
    detect_project(
        cwd=Path("/elsewhere"), projects=None, project_override=str(project_dir_2)
    )

    with open(config_path, "rb") as f:
        data = tomllib.load(f)

    assert data["server"]["host"] == "0.0.0.0"
    assert data["server"]["port"] == 9000
    assert data["search"]["semantic_weight"] == 2.0
    assert data["search"]["keyword_weight"] == 0.5
    assert "projects" not in data


def test_e2e_rebuild_index_cwd_does_not_auto_register(temp_home):
    """
    Integration test: rebuild-index no longer auto-registers an unmatched CWD.
    """
    config_dir = temp_home / ".config" / "mcp-markdown-ragdocs"
    config_dir.mkdir(parents=True)
    config_path = config_dir / "config.toml"

    project_dir = temp_home / "auto-registered-project"
    project_dir.mkdir()

    docs_dir = project_dir / "docs"
    docs_dir.mkdir()
    (docs_dir / "test.md").write_text("# Test Document\n\nTest content.")

    config_file = project_dir / ".mcp-markdown-ragdocs" / "config.toml"
    config_file.parent.mkdir()
    config_file.write_text(
        f"""
[indexing]
documents_path = "{docs_dir}"
"""
    )

    original_cwd = os.getcwd()
    try:
        os.chdir(project_dir)

        runner = CliRunner()
        result = runner.invoke(cli, ["rebuild-index"])

        assert result.exit_code == 0
        assert "Successfully rebuilt index" in result.output
        assert not config_path.exists()

    finally:
        os.chdir(original_cwd)


def test_e2e_mcp_cwd_does_not_auto_register(temp_home):
    """
    Integration test: unmatched CWD detection remains transient for MCP flows too.
    """
    config_dir = temp_home / ".config" / "mcp-markdown-ragdocs"
    config_dir.mkdir(parents=True)
    config_path = config_dir / "config.toml"

    project_dir = temp_home / "mcp-auto-project"
    project_dir.mkdir()

    result = detect_project(cwd=project_dir, projects=None, project_override=None)

    assert result is None
    assert not config_path.exists()
