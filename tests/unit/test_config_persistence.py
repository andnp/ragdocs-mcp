import tomllib
from pathlib import Path

import pytest

from mcp_markdown_ragdocs.config import (
    ProjectConfig,
    _generate_unique_project_name,
    detect_project,
)


@pytest.fixture
def temp_config_home(tmp_path, monkeypatch):
    config_dir = tmp_path / ".config" / "mcp-markdown-ragdocs"
    config_dir.mkdir(parents=True)
    monkeypatch.setenv("HOME", str(tmp_path))
    return config_dir


@pytest.fixture
def sample_projects():
    return [
        ProjectConfig(name="existing-project", path="/home/user/existing"),
        ProjectConfig(name="another-project", path="/home/user/another"),
    ]


def test_generate_unique_project_name_no_conflict():
    """
    Test generating unique project name when no conflicts exist.
    """
    existing_names = ["project-a", "project-b"]
    result = _generate_unique_project_name("my-project", existing_names)
    assert result == "my-project"


def test_generate_unique_project_name_with_conflict():
    """
    Test generating unique project name when name already exists.
    """
    existing_names = ["my-project", "my-project-2", "other"]
    result = _generate_unique_project_name("my-project", existing_names)
    assert result == "my-project-3"


def test_generate_unique_project_name_sanitizes_invalid_chars():
    """
    Test that invalid characters are sanitized to hyphens.
    """
    existing_names = []
    result = _generate_unique_project_name("my project!", existing_names)
    assert result == "my-project"


def test_generate_unique_project_name_multiple_hyphens():
    """
    Test that multiple consecutive hyphens are collapsed to one.
    """
    existing_names = []
    result = _generate_unique_project_name("my---project", existing_names)
    assert result == "my-project"


def test_generate_unique_project_name_strips_leading_trailing_hyphens():
    """
    Test that leading and trailing hyphens are removed.
    """
    existing_names = []
    result = _generate_unique_project_name("-my-project-", existing_names)
    assert result == "my-project"


def test_generate_unique_project_name_fallback_to_project():
    """
    Test that invalid names fall back to 'project'.
    """
    existing_names = []
    result = _generate_unique_project_name("!!!", existing_names)
    assert result == "project"


def test_detect_project_arbitrary_path_is_transient(tmp_path, temp_config_home):
    """
    Test that arbitrary path via --project flag is not persisted.
    """
    config_path = temp_config_home / "config.toml"

    arbitrary_dir = tmp_path / "new-project"
    arbitrary_dir.mkdir()

    result = detect_project(
        cwd=Path("/somewhere/else"), projects=[], project_override=str(arbitrary_dir)
    )

    assert result == "new-project"
    assert not config_path.exists()


def test_detect_project_arbitrary_path_generates_unique_transient_name(
    tmp_path, temp_config_home
):
    """
    Test that arbitrary path generates a unique transient name when conflicts exist.
    """
    config_path = temp_config_home / "config.toml"

    config_path.write_text("""
[[projects]]
name = "my-project"
path = "/existing/my-project"
""")

    arbitrary_dir = tmp_path / "my-project"
    arbitrary_dir.mkdir()

    result = detect_project(
        cwd=Path("/somewhere/else"), projects=None, project_override=str(arbitrary_dir)
    )

    assert result == "my-project-2"

    with open(config_path, "rb") as f:
        data = tomllib.load(f)

    assert len(data["projects"]) == 1
    assert data["projects"][0]["name"] == "my-project"


def test_detect_project_arbitrary_path_invalid_chars(tmp_path, temp_config_home):
    """
    Test that arbitrary path with invalid chars gets sanitized without persistence.
    """
    config_path = temp_config_home / "config.toml"

    arbitrary_dir = tmp_path / "my project!"
    arbitrary_dir.mkdir()

    result = detect_project(
        cwd=Path("/somewhere/else"), projects=[], project_override=str(arbitrary_dir)
    )

    assert result == "my-project"
    assert not config_path.exists()


def test_detect_project_cwd_match_does_not_persist(tmp_path, temp_config_home):
    """
    Test that CWD-based project detection does not persist (already in config).
    """
    config_path = temp_config_home / "config.toml"

    project_dir = tmp_path / "existing"
    project_dir.mkdir()

    config_path.write_text(f"""
[[projects]]
name = "existing"
path = "{project_dir}"
""")

    result = detect_project(cwd=project_dir, projects=None, project_override=None)

    assert result == "existing"

    with open(config_path, "rb") as f:
        data = tomllib.load(f)

    assert len(data["projects"]) == 1

def test_detect_project_cwd_unmatched_does_not_persist(
    tmp_path, temp_config_home
):
    """
    Test that unmatched CWD detection does not auto-persist new projects.
    """
    config_path = temp_config_home / "config.toml"

    project_dir = tmp_path / "unregistered-project"
    project_dir.mkdir()

    result = detect_project(cwd=project_dir, projects=[], project_override=None)

    assert result is None
    assert not config_path.exists()


def test_detect_project_cwd_no_persist_when_already_registered(
    tmp_path, temp_config_home
):
    """
    Test that CWD-based detection does not re-persist when already registered.
    """
    config_path = temp_config_home / "config.toml"

    project_dir = tmp_path / "registered"
    project_dir.mkdir()

    config_path.write_text(f"""
[[projects]]
name = "registered"
path = "{project_dir}"
""")

    result = detect_project(cwd=project_dir, projects=None, project_override=None)

    assert result == "registered"

    with open(config_path, "rb") as f:
        data = tomllib.load(f)

    assert len(data["projects"]) == 1
