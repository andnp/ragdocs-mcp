import pytest
from pathlib import Path
from src.config import Config, IndexingConfig, resolve_index_path


@pytest.fixture
def base_config():
    config = Config()
    config.indexing = IndexingConfig()
    return config


def test_resolve_index_path_local_override(base_config):
    base_config.indexing.index_path = "/custom/index/path"

    result = resolve_index_path(base_config, detected_project="myproject")

    assert result == Path("/custom/index/path").resolve()


def test_resolve_index_path_detected_project(base_config, monkeypatch):
    monkeypatch.delenv("XDG_DATA_HOME", raising=False)

    result = resolve_index_path(base_config, detected_project="myproject")

    expected = Path.home() / ".local/share/mcp-markdown-ragdocs/myproject"
    assert result == expected


def test_resolve_index_path_xdg_data_home(base_config, monkeypatch):
    monkeypatch.setenv("XDG_DATA_HOME", "/custom/data")

    result = resolve_index_path(base_config, detected_project="myproject")

    expected = Path("/custom/data/mcp-markdown-ragdocs/myproject")
    assert result == expected


def test_resolve_index_path_fallback(base_config, monkeypatch):
    """
    When no project is detected, should use global directory with local-{cwd_name} subdirectory.
    """
    monkeypatch.delenv("XDG_DATA_HOME", raising=False)

    result = resolve_index_path(base_config, detected_project=None)

    cwd_name = Path.cwd().name
    expected = Path.home() / ".local/share/mcp-markdown-ragdocs" / f"local-{cwd_name}"
    assert result == expected


def test_resolve_index_path_backward_compat(base_config, monkeypatch):
    """
    When index_path is set to default value in config, should use new global directory behavior.
    The default .index_data/ now triggers global directory usage.
    """
    monkeypatch.delenv("XDG_DATA_HOME", raising=False)
    base_config.indexing.index_path = ".index_data/"

    result = resolve_index_path(base_config, detected_project=None)

    cwd_name = Path.cwd().name
    expected = Path.home() / ".local/share/mcp-markdown-ragdocs" / f"local-{cwd_name}"
    assert result == expected


def test_resolve_index_path_explicit_relative_path(base_config):
    """
    When user explicitly sets a relative path other than default, it should be used.
    """
    base_config.indexing.index_path = "./my_custom_index/"

    result = resolve_index_path(base_config, detected_project=None)

    expected = Path("./my_custom_index/").resolve()
    assert result == expected
