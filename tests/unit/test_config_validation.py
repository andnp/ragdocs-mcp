import pytest
from src.config import ProjectConfig, _validate_projects


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
