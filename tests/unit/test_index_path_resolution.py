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

    result = resolve_index_path(base_config)

    assert result == Path("/custom/index/path").resolve()


def test_resolve_index_path_is_global_even_with_project_context(base_config, monkeypatch):
    monkeypatch.delenv("XDG_DATA_HOME", raising=False)

    result = resolve_index_path(base_config)

    expected = Path.home() / ".local/share/mcp-markdown-ragdocs"
    assert result == expected


def test_resolve_index_path_xdg_data_home(base_config, monkeypatch):
    monkeypatch.setenv("XDG_DATA_HOME", "/custom/data")

    result = resolve_index_path(base_config)

    expected = Path("/custom/data/mcp-markdown-ragdocs")
    assert result == expected


def test_resolve_index_path_default_is_global(base_config, monkeypatch):
    monkeypatch.delenv("XDG_DATA_HOME", raising=False)

    result = resolve_index_path(base_config)

    expected = Path.home() / ".local/share/mcp-markdown-ragdocs"
    assert result == expected


def test_resolve_index_path_backward_compat(base_config, monkeypatch):
    """
    When index_path is set to default value in config, should use new global directory behavior.
    The default .index_data/ now triggers global directory usage.
    """
    monkeypatch.delenv("XDG_DATA_HOME", raising=False)
    base_config.indexing.index_path = ".index_data/"

    result = resolve_index_path(base_config)

    expected = Path.home() / ".local/share/mcp-markdown-ragdocs"
    assert result == expected


def test_resolve_index_path_explicit_relative_path(base_config):
    """
    When user explicitly sets a relative path other than default, it should be used.
    """
    base_config.indexing.index_path = "./my_custom_index/"

    result = resolve_index_path(base_config)

    expected = Path("./my_custom_index/").resolve()
    assert result == expected
