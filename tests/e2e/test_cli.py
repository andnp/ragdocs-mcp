"""
E2E tests for CLI commands (D15).

Tests the command-line interface using Click's CliRunner for isolated
testing. Covers check-config, rebuild-index, run commands, and error handling.
"""

import json
import os
from unittest import mock

import pytest
from click.testing import CliRunner

from src.cli import cli


@pytest.fixture
def runner():
    """
    Create Click CliRunner for CLI testing.

    Provides isolated environment for command execution without
    affecting the actual filesystem or starting real servers.
    """
    return CliRunner()


@pytest.fixture
def config_file(tmp_path):
    """
    Create temporary config file for CLI testing.

    Provides realistic configuration in isolated directory.
    """
    config_path = tmp_path / "config.toml"
    config_path.write_text("""
[server]
host = "127.0.0.1"
port = 8000

[indexing]
documents_path = "docs"
index_path = ".index_data"
recursive = true

[parsers]
"**/*.md" = "MarkdownParser"

[search]
semantic_weight = 1.0
keyword_weight = 1.0
recency_bias = 0.5
rrf_k_constant = 60

[llm]
embedding_model = "local"
llm_provider = "local"
""")
    return config_path


@pytest.fixture
def docs_dir(tmp_path):
    """
    Create test documents directory with sample files.

    Provides markdown files for rebuild-index command testing.
    """
    docs_path = tmp_path / "docs"
    docs_path.mkdir()

    (docs_path / "test1.md").write_text("# Test 1\n\nFirst test document.")
    (docs_path / "test2.md").write_text("# Test 2\n\nSecond test document.")
    (docs_path / "test3.md").write_text("# Test 3\n\nThird test document.")

    return docs_path


def test_check_config_command_prints_valid_config(runner, tmp_path, config_file):
    """
    Test check-config command prints configuration in JSON format.

    Validates that check-config loads configuration successfully and
    outputs all expected sections in readable JSON format.
    """
    # Change to temp directory with config file
    original_cwd = os.getcwd()
    try:
        os.chdir(tmp_path)

        result = runner.invoke(cli, ["check-config"])

        # Verify successful exit
        assert result.exit_code == 0

        # Verify output contains expected text
        assert "Configuration loaded successfully:" in result.output

        # Parse and verify JSON structure
        json_start = result.output.find("{")
        json_end = result.output.rfind("}") + 1
        config_json = result.output[json_start:json_end]
        config_data = json.loads(config_json)

        # Verify all config sections present
        assert "server" in config_data
        assert "indexing" in config_data
        assert "search" in config_data
        assert "llm" in config_data
        assert "parsers" in config_data

        # Verify server section
        assert config_data["server"]["host"] == "127.0.0.1"
        assert config_data["server"]["port"] == 8000

        # Verify indexing section
        assert "documents_path" in config_data["indexing"]
        assert "index_path" in config_data["indexing"]
        assert config_data["indexing"]["recursive"] is True

    finally:
        os.chdir(original_cwd)


def test_rebuild_index_command_rebuilds_indices(runner, tmp_path, config_file, docs_dir):
    """
    Test rebuild-index command processes all markdown files.

    Validates that rebuild-index discovers files, indexes them, and
    persists indices to disk with correct manifest.
    """
    # Setup environment
    original_cwd = os.getcwd()
    try:
        os.chdir(tmp_path)

        # Run rebuild-index command
        result = runner.invoke(cli, ["rebuild-index"])

        # Verify successful exit
        assert result.exit_code == 0

        # Verify success message
        assert "Successfully rebuilt index" in result.output
        assert "documents indexed" in result.output
        assert "3" in result.output  # 3 documents

        # Verify index directories created
        index_path = tmp_path / ".index_data"
        assert index_path.exists()

        # Verify vector index files
        vector_path = index_path / "vector"
        assert vector_path.exists()
        assert (vector_path / "docstore.json").exists()

        # Verify keyword index directory
        keyword_path = index_path / "keyword"
        assert keyword_path.exists()

        # Verify graph store file
        graph_path = index_path / "graph"
        assert graph_path.exists()
        assert (graph_path / "graph.json").exists()

        # Verify manifest file created
        manifest_file = index_path / "index.manifest.json"
        assert manifest_file.exists()

        # Verify manifest contents
        manifest_data = json.loads(manifest_file.read_text())
        assert "spec_version" in manifest_data
        assert "embedding_model" in manifest_data
        assert "parsers" in manifest_data

    finally:
        os.chdir(original_cwd)


def test_run_command_starts_server(runner, tmp_path, config_file):
    """
    Test run command starts uvicorn server with correct parameters.

    Validates that run command loads configuration and calls uvicorn.run
    with appropriate host, port, and factory settings.
    """
    original_cwd = os.getcwd()
    try:
        os.chdir(tmp_path)

        # Mock uvicorn.run to avoid actually starting server
        with mock.patch("src.cli.uvicorn.run") as mock_uvicorn:
            result = runner.invoke(cli, ["run"])

            # Verify successful exit (before uvicorn would start)
            # Note: click exits before uvicorn.run returns
            assert result.exit_code == 0

            # Verify uvicorn.run was called with correct parameters
            mock_uvicorn.assert_called_once()
            call_kwargs = mock_uvicorn.call_args.kwargs

            assert call_kwargs["host"] == "127.0.0.1"
            assert call_kwargs["port"] == 8000
            assert call_kwargs["factory"] is True
            assert "src.server:create_app" in mock_uvicorn.call_args.args

    finally:
        os.chdir(original_cwd)


def test_run_command_with_host_port_overrides(runner, tmp_path, config_file):
    """
    Test run command accepts host and port override flags.

    Validates that command-line arguments override configuration values.
    """
    original_cwd = os.getcwd()
    try:
        os.chdir(tmp_path)

        with mock.patch("src.cli.uvicorn.run") as mock_uvicorn:
            result = runner.invoke(cli, ["run", "--host", "0.0.0.0", "--port", "9000"])

            assert result.exit_code == 0

            # Verify overridden values used
            call_kwargs = mock_uvicorn.call_args.kwargs
            assert call_kwargs["host"] == "0.0.0.0"
            assert call_kwargs["port"] == 9000

    finally:
        os.chdir(original_cwd)


def test_check_config_error_handling_missing_config(runner, tmp_path):
    """
    Test check-config handles missing configuration gracefully.

    Validates that check-config still succeeds with defaults when no
    config file exists (uses default configuration values).
    """
    original_cwd = os.getcwd()
    try:
        # Change to empty directory without config file
        os.chdir(tmp_path)

        result = runner.invoke(cli, ["check-config"])

        # Should succeed with defaults
        assert result.exit_code == 0
        assert "Configuration loaded successfully:" in result.output

        # Parse and verify default values used
        json_start = result.output.find("{")
        json_end = result.output.rfind("}") + 1
        config_json = result.output[json_start:json_end]
        config_data = json.loads(config_json)

        # Verify default values
        assert config_data["server"]["host"] == "127.0.0.1"
        assert config_data["server"]["port"] == 8000
        assert config_data["llm"]["embedding_model"] == "local"

    finally:
        os.chdir(original_cwd)


def test_rebuild_index_error_handling_invalid_config(runner, tmp_path):
    """
    Test rebuild-index handles invalid document paths gracefully.

    Validates error handling when configured documents_path does not exist.
    """
    # Create config pointing to non-existent directory
    config_path = tmp_path / "config.toml"
    config_path.write_text("""
[indexing]
documents_path = "/nonexistent/path"
""")

    original_cwd = os.getcwd()
    try:
        os.chdir(tmp_path)

        result = runner.invoke(cli, ["rebuild-index"])

        # Should complete without crashing (no files found case)
        # Exit code 0 because glob.glob returns empty list for non-existent paths
        assert result.exit_code == 0
        assert "0 documents indexed" in result.output

    finally:
        os.chdir(original_cwd)


def test_run_command_error_handling_config_failure(runner, tmp_path):
    """
    Test run command handles configuration loading failures.

    Validates error handling when config loading encounters errors.
    """
    original_cwd = os.getcwd()
    try:
        os.chdir(tmp_path)

        # Mock load_config to raise exception
        with mock.patch("src.cli.load_config", side_effect=Exception("Config error")):
            result = runner.invoke(cli, ["run"])

            # Verify error exit code (sys.exit(1) in exception handler)
            assert result.exit_code == 1

    finally:
        os.chdir(original_cwd)


def test_cli_group_shows_available_commands(runner):
    """
    Test CLI group displays available commands in help output.

    Validates that help text lists all available commands.
    """
    result = runner.invoke(cli, ["--help"])

    assert result.exit_code == 0

    # Verify all commands listed
    assert "check-config" in result.output
    assert "rebuild-index" in result.output
    assert "run" in result.output
