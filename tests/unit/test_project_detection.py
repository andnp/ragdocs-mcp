import pytest
from pathlib import Path
from src.config import ProjectConfig, detect_project


@pytest.fixture
def sample_projects():
    return [
        ProjectConfig(name="shallow", path="/home/user/shallow"),
        ProjectConfig(name="deep", path="/home/user/shallow/deep"),
        ProjectConfig(name="sibling", path="/home/user/sibling"),
    ]


def test_detect_project_exact_match(sample_projects):
    result = detect_project(
        cwd=Path("/home/user/shallow"),
        projects=sample_projects
    )
    assert result == "shallow"


def test_detect_project_subdirectory(sample_projects):
    result = detect_project(
        cwd=Path("/home/user/shallow/src/lib"),
        projects=sample_projects
    )
    assert result == "shallow"


def test_detect_project_nested_priority(sample_projects):
    result = detect_project(
        cwd=Path("/home/user/shallow/deep/src"),
        projects=sample_projects
    )
    assert result == "deep"


def test_detect_project_no_match(sample_projects):
    result = detect_project(
        cwd=Path("/home/other/path"),
        projects=sample_projects
    )
    assert result is None


def test_detect_project_empty_list():
    result = detect_project(
        cwd=Path("/home/user/anywhere"),
        projects=[]
    )
    assert result is None


def test_detect_project_symlink_resolution(tmp_path, sample_projects):
    real_dir = tmp_path / "real"
    real_dir.mkdir()

    link_dir = tmp_path / "link"
    link_dir.symlink_to(real_dir)

    projects = [ProjectConfig(name="test", path=str(real_dir))]

    result = detect_project(cwd=link_dir, projects=projects)
    assert result == "test"


def test_project_config_validates_name():
    with pytest.raises(ValueError, match="Invalid project name"):
        ProjectConfig(name="invalid name!", path="/home/user/test")


def test_project_config_validates_absolute_path():
    with pytest.raises(ValueError, match="must be absolute"):
        ProjectConfig(name="test", path="relative/path")


def test_project_config_expands_tilde():
    project = ProjectConfig(name="test", path="~/test")
    assert project.path.startswith("/")
    assert "~" not in project.path


def test_detect_project_override_by_name(sample_projects):
    """
    Test manual project override using --project flag with project name.
    """
    result = detect_project(
        cwd=Path("/home/user/other"),
        projects=sample_projects,
        project_override="deep"
    )
    assert result == "deep"


def test_detect_project_override_by_path(sample_projects):
    """
    Test manual project override using --project flag with absolute path.
    """
    result = detect_project(
        cwd=Path("/home/user/other"),
        projects=sample_projects,
        project_override="/home/user/shallow/deep"
    )
    assert result == "deep"


def test_detect_project_override_not_found(sample_projects):
    """
    Test that invalid project override returns None (with warning logged).
    """
    result = detect_project(
        cwd=Path("/home/user/shallow"),
        projects=sample_projects,
        project_override="nonexistent"
    )
    assert result is None


def test_detect_project_override_takes_precedence(sample_projects):
    """
    Test that project override takes precedence over CWD detection.
    """
    result = detect_project(
        cwd=Path("/home/user/shallow"),
        projects=sample_projects,
        project_override="sibling"
    )
    assert result == "sibling"


def test_detect_project_override_arbitrary_path(tmp_path, sample_projects):
    """
    Test that --project flag accepts arbitrary absolute paths not in registry.
    """
    arbitrary_dir = tmp_path / "arbitrary_project"
    arbitrary_dir.mkdir()

    result = detect_project(
        cwd=Path("/home/user/shallow"),
        projects=sample_projects,
        project_override=str(arbitrary_dir)
    )

    assert result == "arbitrary_project"
