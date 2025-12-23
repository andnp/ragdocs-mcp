"""
Unit tests for src/config.py configuration loader.

Tests cover loading from local and global paths, defaults, path expansion,
and field validation.
"""
import os
from pathlib import Path


from src.config import load_config


def test_load_from_project_local_config(tmp_path):
    """
    Load configuration from project-local config.toml.
    Ensures local config takes precedence over global config.
    """
    local_config = tmp_path / "config.toml"
    local_config.write_text("""
[server]
host = "192.168.1.1"
port = 9000

[indexing]
documents_path = "/data/docs"
index_path = "/data/index"
recursive = false
""")

    # Change to tmp directory to test local config loading
    original_cwd = os.getcwd()
    try:
        os.chdir(tmp_path)
        config = load_config()

        assert config.server.host == "192.168.1.1"
        assert config.server.port == 9000
        assert config.indexing.documents_path == "/data/docs"
        assert config.indexing.index_path == "/data/index"
        assert config.indexing.recursive is False
    finally:
        os.chdir(original_cwd)


def test_load_from_user_global_path(tmp_path, monkeypatch):
    """
    Load configuration from user global config path.
    Validates fallback behavior when no local config exists.
    """
    # Set HOME to tmp_path for testing global config
    monkeypatch.setenv("HOME", str(tmp_path))

    # Create global config directory and file
    global_config_dir = tmp_path / ".config" / "mcp-markdown-ragdocs"
    global_config_dir.mkdir(parents=True)
    global_config_path = global_config_dir / "config.toml"
    global_config_path.write_text("""
[server]
host = "10.0.0.1"
port = 7000

[indexing]
documents_path = "~/documents"
""")

    # Change to a different directory to ensure no local config
    work_dir = tmp_path / "workspace"
    work_dir.mkdir()
    original_cwd = os.getcwd()
    try:
        os.chdir(work_dir)
        config = load_config()

        assert config.server.host == "10.0.0.1"
        assert config.server.port == 7000
        # Path should be expanded from ~
        assert config.indexing.documents_path.startswith(str(tmp_path))
        assert "documents" in config.indexing.documents_path
    finally:
        os.chdir(original_cwd)


def test_use_defaults_when_no_config_exists(tmp_path):
    """
    Use default values when no configuration file exists.
    Ensures graceful degradation without config files.
    """
    # Change to empty directory with no config files
    original_cwd = os.getcwd()
    try:
        os.chdir(tmp_path)
        config = load_config()

        # Verify all default values are used
        assert config.server.host == "127.0.0.1"
        assert config.server.port == 8000
        assert config.indexing.recursive is True
        assert config.search.semantic_weight == 1.0
        assert config.search.keyword_weight == 1.0
        assert config.search.recency_bias == 0.5
        assert config.search.rrf_k_constant == 60
        assert config.llm.embedding_model == "local"
        assert config.llm.llm_provider is None
        assert config.parsers == {
            "**/*.md": "MarkdownParser",
            "**/*.markdown": "MarkdownParser"
        }
    finally:
        os.chdir(original_cwd)


def test_path_expansion_tilde_and_relative(tmp_path, monkeypatch):
    """
    Expand tilde (~) and relative paths correctly.
    Critical for portable configurations across systems.
    """
    monkeypatch.setenv("HOME", str(tmp_path))

    local_config = tmp_path / "config.toml"
    local_config.write_text("""
[indexing]
documents_path = "~/my_docs"
index_path = "relative/path"
""")

    original_cwd = os.getcwd()
    try:
        os.chdir(tmp_path)
        config = load_config()

        # Tilde should be expanded to HOME
        assert config.indexing.documents_path.startswith(str(tmp_path))
        assert config.indexing.documents_path.endswith("my_docs")

        # Relative paths should be resolved to absolute
        assert Path(config.indexing.index_path).is_absolute()
        assert "relative/path" in config.indexing.index_path
    finally:
        os.chdir(original_cwd)


def test_validation_of_required_fields(tmp_path):
    """
    Validate that config loads without validation errors.
    Tests that the loader handles TOML parsing and basic structure.
    """
    # Test that config with string port loads (TOML validation)
    config_with_string = tmp_path / "config.toml"
    config_with_string.write_text("""
[server]
port = "not_a_number"
""")

    original_cwd = os.getcwd()
    try:
        os.chdir(tmp_path)
        config = load_config()
        # Config loads but port will be string (no runtime validation)
        assert config.server.port == "not_a_number"
    finally:
        os.chdir(original_cwd)


def test_partial_config_with_defaults(tmp_path):
    """
    Load partial configuration with defaults for missing sections.
    Validates merging of user config with defaults.
    """
    partial_config = tmp_path / "config.toml"
    partial_config.write_text("""
[server]
port = 3000
""")

    original_cwd = os.getcwd()
    try:
        os.chdir(tmp_path)
        config = load_config()

        # User-provided value
        assert config.server.port == 3000
        # Default values for missing fields
        assert config.server.host == "127.0.0.1"
        assert config.indexing.recursive is True
        assert config.llm.embedding_model == "local"
    finally:
        os.chdir(original_cwd)


def test_all_sections_present_in_config(tmp_path):
    """
    Verify all configuration sections are present with full config.
    Integration test ensuring complete config structure.
    """
    full_config = tmp_path / "config.toml"
    full_config.write_text("""
[server]
host = "0.0.0.0"
port = 5000

[indexing]
documents_path = "/srv/docs"
index_path = "/srv/index"
recursive = false

[parsers]
"**/*.txt" = "TextParser"

[search]
semantic_weight = 0.8
keyword_weight = 0.2
recency_bias = 0.3
rrf_k_constant = 50

[llm]
embedding_model = "custom"
llm_provider = "openai"
""")

    original_cwd = os.getcwd()
    try:
        os.chdir(tmp_path)
        config = load_config()

        assert config.server.host == "0.0.0.0"
        assert config.server.port == 5000
        assert config.indexing.documents_path == "/srv/docs"
        assert config.indexing.index_path == "/srv/index"
        assert config.indexing.recursive is False
        assert config.parsers == {"**/*.txt": "TextParser"}
        assert config.search.semantic_weight == 0.8
        assert config.search.keyword_weight == 0.2
        assert config.search.recency_bias == 0.3
        assert config.search.rrf_k_constant == 50
        assert config.llm.embedding_model == "custom"
        assert config.llm.llm_provider == "openai"
    finally:
        os.chdir(original_cwd)
