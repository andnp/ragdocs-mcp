import pytest
import tomllib
from pathlib import Path
import src.config
from src.config import (
    ProjectConfig,
    detect_project,
    persist_project_to_config,
    _generate_unique_project_name,
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


def test_persist_project_to_config_creates_config_dir(temp_config_home):
    """
    Test that config directory is created if it doesn't exist.
    """
    config_path = temp_config_home / "config.toml"
    assert not config_path.exists()

    persist_project_to_config("test-project", "/path/to/test")

    assert config_path.exists()


def test_persist_project_to_config_creates_new_file(temp_config_home):
    """
    Test persisting project when config file doesn't exist.
    """
    config_path = temp_config_home / "config.toml"

    persist_project_to_config("test-project", "/path/to/test")

    with open(config_path, "rb") as f:
        data = tomllib.load(f)

    assert "projects" in data
    assert len(data["projects"]) == 1
    assert data["projects"][0]["name"] == "test-project"
    assert data["projects"][0]["path"] == "/path/to/test"


def test_persist_project_to_config_appends_to_existing(temp_config_home):
    """
    Test persisting project appends to existing config.
    """
    config_path = temp_config_home / "config.toml"

    config_path.write_text("""
[[projects]]
name = "existing"
path = "/path/to/existing"
""")

    persist_project_to_config("new-project", "/path/to/new")

    with open(config_path, "rb") as f:
        data = tomllib.load(f)

    assert len(data["projects"]) == 2
    assert data["projects"][0]["name"] == "existing"
    assert data["projects"][1]["name"] == "new-project"


def test_persist_project_to_config_prevents_duplicate_names(temp_config_home):
    """
    Test that duplicate project names are not persisted.
    """
    config_path = temp_config_home / "config.toml"

    config_path.write_text("""
[[projects]]
name = "test-project"
path = "/path/to/existing"
""")

    persist_project_to_config("test-project", "/path/to/new")

    with open(config_path, "rb") as f:
        data = tomllib.load(f)

    assert len(data["projects"]) == 1
    assert data["projects"][0]["path"] == "/path/to/existing"


def test_persist_project_to_config_prevents_duplicate_paths(temp_config_home):
    """
    Test that duplicate project paths are not persisted.
    """
    config_path = temp_config_home / "config.toml"

    config_path.write_text("""
[[projects]]
name = "existing"
path = "/path/to/test"
""")

    persist_project_to_config("new-name", "/path/to/test")

    with open(config_path, "rb") as f:
        data = tomllib.load(f)

    assert len(data["projects"]) == 1
    assert data["projects"][0]["name"] == "existing"


def test_persist_project_to_config_preserves_other_config(temp_config_home):
    """
    Test that persisting project preserves other config sections.
    """
    config_path = temp_config_home / "config.toml"

    config_path.write_text("""
[server]
host = "0.0.0.0"
port = 9000

[llm]
embedding_model = "custom"
""")

    persist_project_to_config("test-project", "/path/to/test")

    with open(config_path, "rb") as f:
        data = tomllib.load(f)

    assert data["server"]["host"] == "0.0.0.0"
    assert data["server"]["port"] == 9000
    assert data["llm"]["embedding_model"] == "custom"
    assert len(data["projects"]) == 1


def test_detect_project_arbitrary_path_persists(tmp_path, temp_config_home):
    """
    Test that arbitrary path via --project flag gets persisted.
    """
    config_path = temp_config_home / "config.toml"

    arbitrary_dir = tmp_path / "new-project"
    arbitrary_dir.mkdir()

    result = detect_project(
        cwd=Path("/somewhere/else"),
        projects=[],
        project_override=str(arbitrary_dir)
    )

    assert result == "new-project"
    assert config_path.exists()

    with open(config_path, "rb") as f:
        data = tomllib.load(f)

    assert len(data["projects"]) == 1
    assert data["projects"][0]["name"] == "new-project"
    assert data["projects"][0]["path"] == str(arbitrary_dir)


def test_detect_project_arbitrary_path_generates_unique_name(tmp_path, temp_config_home):
    """
    Test that arbitrary path generates unique name when conflicts exist.
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
        cwd=Path("/somewhere/else"),
        projects=None,
        project_override=str(arbitrary_dir)
    )

    assert result == "my-project-2"

    with open(config_path, "rb") as f:
        data = tomllib.load(f)

    assert len(data["projects"]) == 2
    assert data["projects"][1]["name"] == "my-project-2"


def test_detect_project_arbitrary_path_invalid_chars(tmp_path, temp_config_home):
    """
    Test that arbitrary path with invalid chars gets sanitized.
    """
    config_path = temp_config_home / "config.toml"

    arbitrary_dir = tmp_path / "my project!"
    arbitrary_dir.mkdir()

    result = detect_project(
        cwd=Path("/somewhere/else"),
        projects=[],
        project_override=str(arbitrary_dir)
    )

    assert result == "my-project"

    with open(config_path, "rb") as f:
        data = tomllib.load(f)

    assert data["projects"][0]["name"] == "my-project"


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

    result = detect_project(
        cwd=project_dir,
        projects=None,
        project_override=None
    )

    assert result == "existing"

    with open(config_path, "rb") as f:
        data = tomllib.load(f)

    assert len(data["projects"]) == 1


def test_persist_project_atomic_write_failure_handling(temp_config_home, monkeypatch):
    """
    Test that atomic write failures are handled gracefully.
    """
    config_path = temp_config_home / "config.toml"
    config_path.write_text("""
[[projects]]
name = "existing"
path = "/path/to/existing"
""")

    def failing_replace(self, target):
        raise OSError("Simulated write failure")

    monkeypatch.setattr(Path, "replace", failing_replace)

    try:
        persist_project_to_config("new-project", "/path/to/new")
    except OSError:
        pass

    with open(config_path, "rb") as f:
        data = tomllib.load(f)

    assert len(data["projects"]) == 1
    assert data["projects"][0]["name"] == "existing"


def test_detect_project_persistence_failure_returns_name(tmp_path, temp_config_home, monkeypatch):
    """
    Test that detect_project returns project name even if persistence fails.
    """
    arbitrary_dir = tmp_path / "test-project"
    arbitrary_dir.mkdir()

    def failing_persist(name, path):
        raise OSError("Simulated persistence failure")

    monkeypatch.setattr(src.config, "persist_project_to_config", failing_persist)

    result = detect_project(
        cwd=Path("/somewhere/else"),
        projects=[],
        project_override=str(arbitrary_dir)
    )

    assert result == "test-project"
