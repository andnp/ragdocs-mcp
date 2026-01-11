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
    config_dir = tmp_path / ".mcp-markdown-ragdocs"
    config_dir.mkdir()
    config_path = config_dir / "config.toml"
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
    Test check-config command prints configuration in table format.

    Validates that check-config loads configuration successfully and
    outputs all expected sections in readable table format.
    """
    original_cwd = os.getcwd()
    try:
        os.chdir(tmp_path)

        result = runner.invoke(cli, ["check-config"])

        assert result.exit_code == 0

        assert "Configuration" in result.output
        assert "Server Host" in result.output
        assert "Server Port" in result.output
        assert "Documents Path" in result.output
        assert "Index Path" in result.output
        assert "✅ Configuration is valid" in result.output

    finally:
        os.chdir(original_cwd)


def test_rebuild_index_command_rebuilds_indices(runner, tmp_path, config_file, docs_dir):
    """
    Test rebuild-index command processes all markdown files.

    Validates that rebuild-index discovers files, indexes them, and
    reports successful completion. File verification done in unit tests.
    """
    original_cwd = os.getcwd()
    try:
        os.chdir(tmp_path)

        result = runner.invoke(cli, ["rebuild-index"])

        assert result.exit_code == 0

        assert "Successfully rebuilt index" in result.output
        assert "documents indexed" in result.output
        assert "3" in result.output

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

        assert result.exit_code == 0
        assert "Configuration" in result.output
        assert "✅ Configuration is valid" in result.output

    finally:
        os.chdir(original_cwd)


def test_rebuild_index_error_handling_invalid_config(runner, tmp_path):
    """
    Test rebuild-index handles invalid document paths gracefully.

    Validates error handling when configured documents_path does not exist.
    """
    # Create config pointing to non-existent directory
    config_dir = tmp_path / ".mcp-markdown-ragdocs"
    config_dir.mkdir()
    config_path = config_dir / "config.toml"
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


# =============================================================================
# Git Commit Indexing Tests
# =============================================================================


def test_rebuild_index_with_git_enabled_indexes_commits(runner, tmp_path, config_file, docs_dir):
    """
    Test rebuild-index indexes git commits when git_indexing enabled.

    Validates that commits are discovered, indexed, and reported.
    """
    import subprocess

    original_cwd = os.getcwd()
    try:
        os.chdir(tmp_path)

        # Initialize git repo with commits
        subprocess.run(["git", "init"], cwd=docs_dir, check=True, capture_output=True)
        subprocess.run(["git", "config", "user.name", "Test User"], cwd=docs_dir, check=True, capture_output=True)
        subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=docs_dir, check=True, capture_output=True)
        subprocess.run(["git", "add", "."], cwd=docs_dir, check=True, capture_output=True)
        subprocess.run(["git", "commit", "-m", "Initial commit"], cwd=docs_dir, check=True, capture_output=True)

        # Add another commit
        (docs_dir / "test4.md").write_text("# Test 4\n\nFourth test document.")
        subprocess.run(["git", "add", "test4.md"], cwd=docs_dir, check=True, capture_output=True)
        subprocess.run(["git", "commit", "-m", "Add test4"], cwd=docs_dir, check=True, capture_output=True)

        # Ensure git_indexing is enabled in config
        config_dir = tmp_path / ".mcp-markdown-ragdocs"
        config_path = config_dir / "config.toml"
        config_content = config_path.read_text()
        if "[git_indexing]" not in config_content:
            config_content += "\n[git_indexing]\nenabled = true\n"
            config_path.write_text(config_content)

        result = runner.invoke(cli, ["rebuild-index"])

        assert result.exit_code == 0
        assert "Successfully rebuilt index" in result.output
        assert "4 documents indexed" in result.output  # 3 original + test4.md
        assert "Indexing git commits" in result.output
        assert "Successfully indexed" in result.output
        assert "git commits" in result.output
        assert "repositories" in result.output

    finally:
        os.chdir(original_cwd)


def test_rebuild_index_with_git_disabled_skips_commits(runner, tmp_path, config_file, docs_dir):
    """
    Test rebuild-index skips git indexing when disabled in config.

    Validates that git phase is entirely skipped.
    """
    import subprocess

    original_cwd = os.getcwd()
    try:
        os.chdir(tmp_path)

        # Initialize git repo
        subprocess.run(["git", "init"], cwd=docs_dir, check=True, capture_output=True)
        subprocess.run(["git", "config", "user.name", "Test User"], cwd=docs_dir, check=True, capture_output=True)
        subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=docs_dir, check=True, capture_output=True)
        subprocess.run(["git", "add", "."], cwd=docs_dir, check=True, capture_output=True)
        subprocess.run(["git", "commit", "-m", "Initial commit"], cwd=docs_dir, check=True, capture_output=True)

        # Disable git_indexing in config
        config_dir = tmp_path / ".mcp-markdown-ragdocs"
        config_path = config_dir / "config.toml"
        config_content = config_path.read_text()
        config_content += "\n[git_indexing]\nenabled = false\n"
        config_path.write_text(config_content)

        result = runner.invoke(cli, ["rebuild-index"])

        assert result.exit_code == 0
        assert "Successfully rebuilt index" in result.output
        # Should NOT mention git commits
        assert "git commits" not in result.output.lower() or "skipping git" in result.output.lower()

    finally:
        os.chdir(original_cwd)


def test_rebuild_index_no_git_repos_informational_message(runner, tmp_path, config_file):
    """
    Test rebuild-index handles absence of git repositories gracefully.

    Validates informational message when no .git directories found.
    """
    original_cwd = os.getcwd()
    try:
        os.chdir(tmp_path)

        # Create docs without git
        docs_path = tmp_path / "docs"
        docs_path.mkdir()
        (docs_path / "test.md").write_text("# Test\n\nTest document.")

        # Ensure git_indexing is enabled
        config_dir = tmp_path / ".mcp-markdown-ragdocs"
        config_path = config_dir / "config.toml"
        config_content = config_path.read_text()
        if "[git_indexing]" not in config_content:
            config_content += "\n[git_indexing]\nenabled = true\n"
            config_path.write_text(config_content)

        result = runner.invoke(cli, ["rebuild-index"])

        assert result.exit_code == 0
        assert "Successfully rebuilt index" in result.output
        assert "1 documents indexed" in result.output
        assert "No git repositories found" in result.output

    finally:
        os.chdir(original_cwd)


def test_rebuild_index_git_error_handling_non_fatal(runner, tmp_path, config_file, docs_dir):
    """
    Test rebuild-index handles git errors without failing entire rebuild.

    Validates that git indexing errors are logged but don't block document indexing.
    """
    original_cwd = os.getcwd()
    try:
        os.chdir(tmp_path)

        # Create corrupted git directory
        git_dir = docs_dir / ".git"
        git_dir.mkdir()
        (git_dir / "HEAD").write_text("invalid content")

        # Ensure git_indexing is enabled
        config_dir = tmp_path / ".mcp-markdown-ragdocs"
        config_path = config_dir / "config.toml"
        config_content = config_path.read_text()
        if "[git_indexing]" not in config_content:
            config_content += "\n[git_indexing]\nenabled = true\n"
            config_path.write_text(config_content)

        result = runner.invoke(cli, ["rebuild-index"])

        # Should still succeed - git error is non-fatal
        assert result.exit_code == 0
        assert "Successfully rebuilt index" in result.output
        assert "3 documents indexed" in result.output
        # May have warning about git failure
        # But document indexing succeeded

    finally:
        os.chdir(original_cwd)


# =============================================================================
# Concept Vocabulary Tests
# =============================================================================


def test_rebuild_index_builds_vocabulary_when_enabled(runner, tmp_path, config_file, docs_dir):
    """
    Test rebuild-index builds concept vocabulary when query_expansion_enabled.

    Validates vocabulary is built and persisted.
    """
    original_cwd = os.getcwd()
    try:
        os.chdir(tmp_path)

        # Ensure query_expansion is enabled
        config_dir = tmp_path / ".mcp-markdown-ragdocs"
        config_path = config_dir / "config.toml"
        config_content = config_path.read_text()
        if "[search]" in config_content:
            # Add query_expansion_enabled to existing search section
            config_content = config_content.replace(
                "[search]",
                "[search]\nquery_expansion_enabled = true"
            )
        else:
            config_content += "\n[search]\nquery_expansion_enabled = true\n"
        config_path.write_text(config_content)

        result = runner.invoke(cli, ["rebuild-index"])

        assert result.exit_code == 0
        assert "Successfully rebuilt index" in result.output
        assert "Building concept vocabulary" in result.output
        assert "Successfully built concept vocabulary" in result.output
        assert "terms" in result.output

    finally:
        os.chdir(original_cwd)


def test_rebuild_index_skips_vocabulary_when_disabled(runner, tmp_path, config_file, docs_dir):
    """
    Test rebuild-index skips vocabulary building when query_expansion_enabled is false.

    Validates vocabulary phase is skipped entirely.
    """
    original_cwd = os.getcwd()
    try:
        os.chdir(tmp_path)

        # Disable query_expansion
        config_dir = tmp_path / ".mcp-markdown-ragdocs"
        config_path = config_dir / "config.toml"
        config_content = config_path.read_text()
        if "[search]" in config_content:
            config_content = config_content.replace(
                "[search]",
                "[search]\nquery_expansion_enabled = false"
            )
        else:
            config_content += "\n[search]\nquery_expansion_enabled = false\n"
        config_path.write_text(config_content)

        result = runner.invoke(cli, ["rebuild-index"])

        assert result.exit_code == 0
        assert "Successfully rebuilt index" in result.output
        # Should NOT mention vocabulary building
        assert "Building concept vocabulary" not in result.output

    finally:
        os.chdir(original_cwd)


def test_rebuild_index_vocabulary_error_handling_non_fatal(runner, tmp_path, config_file, docs_dir):
    """
    Test rebuild-index handles vocabulary errors without failing entire rebuild.

    Validates that vocabulary errors are logged but don't block document indexing.
    """
    original_cwd = os.getcwd()
    try:
        os.chdir(tmp_path)

        # Ensure query_expansion is enabled
        config_dir = tmp_path / ".mcp-markdown-ragdocs"
        config_path = config_dir / "config.toml"
        config_content = config_path.read_text()
        if "[search]" in config_content:
            config_content = config_content.replace(
                "[search]",
                "[search]\nquery_expansion_enabled = true"
            )
        else:
            config_content += "\n[search]\nquery_expansion_enabled = true\n"
        config_path.write_text(config_content)

        # Mock build_concept_vocabulary to raise exception
        with mock.patch("src.indices.vector.VectorIndex.build_concept_vocabulary", side_effect=Exception("Vocabulary error")):
            result = runner.invoke(cli, ["rebuild-index"])

            # Should still succeed - vocabulary error is non-fatal
            assert result.exit_code == 0
            assert "Successfully rebuilt index" in result.output
            assert "3 documents indexed" in result.output
            assert "Concept vocabulary building failed" in result.output
            assert "Vocabulary error" in result.output

    finally:
        os.chdir(original_cwd)


def test_rebuild_index_both_git_and_vocabulary_enabled(runner, tmp_path, config_file, docs_dir):
    """
    Test rebuild-index executes both git indexing and vocabulary building.

    Validates both phases execute when enabled and artifacts exist.
    """
    import subprocess

    original_cwd = os.getcwd()
    try:
        os.chdir(tmp_path)

        # Initialize git repo with commits
        subprocess.run(["git", "init"], cwd=docs_dir, check=True, capture_output=True)
        subprocess.run(["git", "config", "user.name", "Test User"], cwd=docs_dir, check=True, capture_output=True)
        subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=docs_dir, check=True, capture_output=True)
        subprocess.run(["git", "add", "."], cwd=docs_dir, check=True, capture_output=True)
        subprocess.run(["git", "commit", "-m", "Initial commit"], cwd=docs_dir, check=True, capture_output=True)

        # Enable both git_indexing and query_expansion
        config_dir = tmp_path / ".mcp-markdown-ragdocs"
        config_path = config_dir / "config.toml"
        config_content = config_path.read_text()
        if "[git_indexing]" not in config_content:
            config_content += "\n[git_indexing]\nenabled = true\n"
        if "[search]" in config_content:
            config_content = config_content.replace(
                "[search]",
                "[search]\nquery_expansion_enabled = true"
            )
        else:
            config_content += "\n[search]\nquery_expansion_enabled = true\n"
        config_path.write_text(config_content)

        result = runner.invoke(cli, ["rebuild-index"])

        assert result.exit_code == 0
        assert "Successfully rebuilt index" in result.output
        assert "documents indexed" in result.output

        # Verify git indexing happened
        assert "Indexing git commits" in result.output or "Successfully indexed" in result.output
        assert "git commits" in result.output

        # Verify vocabulary building happened
        assert "Building concept vocabulary" in result.output
        assert "Successfully built concept vocabulary" in result.output

    finally:
        os.chdir(original_cwd)
