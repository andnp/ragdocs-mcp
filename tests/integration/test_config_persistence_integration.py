import pytest
import tomllib
from pathlib import Path
from src.config import detect_project


@pytest.fixture
def temp_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    return tmp_path


def test_e2e_arbitrary_path_persists_and_loads(temp_home):
    """
    End-to-end test: arbitrary path gets persisted and can be loaded in future sessions.
    """
    config_dir = temp_home / ".config" / "mcp-markdown-ragdocs"
    config_path = config_dir / "config.toml"

    project_dir = temp_home / "my-awesome-project"
    project_dir.mkdir(parents=True)

    result = detect_project(
        cwd=Path("/somewhere/else"),
        projects=None,
        project_override=str(project_dir)
    )

    assert result == "my-awesome-project"
    assert config_path.exists()

    with open(config_path, "rb") as f:
        data = tomllib.load(f)

    assert len(data["projects"]) == 1
    assert data["projects"][0]["name"] == "my-awesome-project"
    assert data["projects"][0]["path"] == str(project_dir)

    result_again = detect_project(
        cwd=Path("/somewhere/else"),
        projects=None,
        project_override="my-awesome-project"
    )

    assert result_again == "my-awesome-project"


def test_e2e_multiple_arbitrary_paths_with_conflicts(temp_home):
    """
    End-to-end test: multiple arbitrary paths with same dir name get unique names.
    """
    project_dir_1 = temp_home / "projects" / "frontend"
    project_dir_1.mkdir(parents=True)

    project_dir_2 = temp_home / "work" / "frontend"
    project_dir_2.mkdir(parents=True)

    project_dir_3 = temp_home / "personal" / "frontend"
    project_dir_3.mkdir(parents=True)

    result_1 = detect_project(
        cwd=Path("/somewhere"),
        projects=None,
        project_override=str(project_dir_1)
    )
    assert result_1 == "frontend"

    result_2 = detect_project(
        cwd=Path("/somewhere"),
        projects=None,
        project_override=str(project_dir_2)
    )
    assert result_2 == "frontend-2"

    result_3 = detect_project(
        cwd=Path("/somewhere"),
        projects=None,
        project_override=str(project_dir_3)
    )
    assert result_3 == "frontend-3"

    config_path = temp_home / ".config" / "mcp-markdown-ragdocs" / "config.toml"
    with open(config_path, "rb") as f:
        data = tomllib.load(f)

    assert len(data["projects"]) == 3
    assert data["projects"][0]["name"] == "frontend"
    assert data["projects"][1]["name"] == "frontend-2"
    assert data["projects"][2]["name"] == "frontend-3"


def test_e2e_persisted_project_available_in_next_detect(temp_home):
    """
    End-to-end test: persisted project is immediately available for detection.
    """
    project_dir = temp_home / "test-project"
    project_dir.mkdir(parents=True)

    subdirectory = project_dir / "src" / "lib"
    subdirectory.mkdir(parents=True)

    result = detect_project(
        cwd=Path("/somewhere"),
        projects=None,
        project_override=str(project_dir)
    )
    assert result == "test-project"

    result_by_cwd = detect_project(
        cwd=subdirectory,
        projects=None,
        project_override=None
    )
    assert result_by_cwd == "test-project"

    result_by_name = detect_project(
        cwd=Path("/other/location"),
        projects=None,
        project_override="test-project"
    )
    assert result_by_name == "test-project"


def test_e2e_config_preserved_across_multiple_persists(temp_home):
    """
    End-to-end test: other config sections preserved when persisting projects.
    """
    config_dir = temp_home / ".config" / "mcp-markdown-ragdocs"
    config_dir.mkdir(parents=True)
    config_path = config_dir / "config.toml"

    config_path.write_text("""
[server]
host = "0.0.0.0"
port = 9000

[search]
semantic_weight = 2.0
keyword_weight = 0.5
""")

    project_dir_1 = temp_home / "project-1"
    project_dir_1.mkdir()
    detect_project(cwd=Path("/elsewhere"), projects=None, project_override=str(project_dir_1))

    project_dir_2 = temp_home / "project-2"
    project_dir_2.mkdir()
    detect_project(cwd=Path("/elsewhere"), projects=None, project_override=str(project_dir_2))

    with open(config_path, "rb") as f:
        data = tomllib.load(f)

    assert data["server"]["host"] == "0.0.0.0"
    assert data["server"]["port"] == 9000
    assert data["search"]["semantic_weight"] == 2.0
    assert data["search"]["keyword_weight"] == 0.5
    assert len(data["projects"]) == 2
