import pytest
from src.config import ProjectConfig, _validate_projects, get_project_root_warnings


def test_validate_projects_duplicate_names():
    projects = [
        ProjectConfig(name="project", path="/home/user/project1"),
        ProjectConfig(name="project", path="/home/user/project2"),
    ]

    with pytest.raises(ValueError, match="Duplicate project names"):
        _validate_projects(projects)


def test_validate_projects_duplicate_paths():
    projects = [
        ProjectConfig(name="project1", path="/home/user/project"),
        ProjectConfig(name="project2", path="/home/user/project"),
    ]

    with pytest.raises(ValueError, match="Duplicate project paths"):
        _validate_projects(projects)


def test_validate_projects_unique():
    projects = [
        ProjectConfig(name="project1", path="/home/user/project1"),
        ProjectConfig(name="project2", path="/home/user/project2"),
    ]

    _validate_projects(projects)


def test_get_project_root_warnings_home_directory(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))

    projects = [ProjectConfig(name="home", path=str(tmp_path))]

    warnings = get_project_root_warnings(projects)

    assert warnings == [
        f"Project 'home' path '{tmp_path}' is the current user's home directory."
    ]


def test_get_project_root_warnings_filesystem_root():
    projects = [ProjectConfig(name="root", path="/")]

    warnings = get_project_root_warnings(projects)

    assert warnings == ["Project 'root' path '/' is the filesystem root."]


def test_get_project_root_warnings_for_parent_project(tmp_path):
    parent = tmp_path / "workspace"
    child = parent / "service"
    child.mkdir(parents=True)

    projects = [
        ProjectConfig(name="workspace", path=str(parent)),
        ProjectConfig(name="service", path=str(child)),
    ]

    warnings = get_project_root_warnings(projects)

    assert warnings == [
        f"Project 'workspace' path '{parent}' contains other registered project roots: service."
    ]
