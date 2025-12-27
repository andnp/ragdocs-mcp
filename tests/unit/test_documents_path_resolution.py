import pytest
from pathlib import Path
from src.config import Config, IndexingConfig, ProjectConfig, resolve_documents_path


@pytest.fixture
def sample_projects():
    return [
        ProjectConfig(name="project1", path="/home/user/projects/project1"),
        ProjectConfig(name="project2", path="/home/user/projects/project2"),
    ]


@pytest.fixture
def default_config():
    return Config()


def test_resolve_documents_path_relative_with_project(default_config, sample_projects):
    """When project detected, ALWAYS use project path (ignore documents_path)."""
    result = resolve_documents_path(default_config, "project1", sample_projects)
    assert result == "/home/user/projects/project1"


def test_resolve_documents_path_relative_subdir_with_project(sample_projects):
    """When project detected, ALWAYS use project path (ignore documents_path, even if it's a subdir)."""
    config = Config(indexing=IndexingConfig(documents_path="docs"))
    result = resolve_documents_path(config, "project1", sample_projects)
    # New behavior: project path is ALWAYS the documents root, documents_path is ignored
    assert result == "/home/user/projects/project1"


def test_resolve_documents_path_absolute():
    """When documents_path is absolute, use it as-is regardless of project."""
    config = Config(indexing=IndexingConfig(documents_path="/absolute/path/docs"))
    result = resolve_documents_path(config, "project1", [])
    assert result == "/absolute/path/docs"


def test_resolve_documents_path_no_project(default_config, tmp_path):
    """When no project detected, resolve relative to CWD."""
    import os
    original_cwd = Path.cwd()
    try:
        os.chdir(tmp_path)
        result = resolve_documents_path(default_config, None, [])
        assert result == str(tmp_path)
    finally:
        os.chdir(original_cwd)


def test_resolve_documents_path_project_not_in_list(default_config, sample_projects):
    """When detected project not found in projects list, resolve relative to CWD."""
    import os
    original_cwd = Path.cwd()
    try:
        temp_dir = Path("/tmp")
        os.chdir(temp_dir)
        result = resolve_documents_path(default_config, "nonexistent", sample_projects)
        assert result == str(temp_dir)
    finally:
        os.chdir(original_cwd)


def test_resolve_documents_path_expanduser():
    """Tilde expansion should work."""
    config = Config(indexing=IndexingConfig(documents_path="~/docs"))
    result = resolve_documents_path(config, None, [])
    assert result == str(Path.home() / "docs")
