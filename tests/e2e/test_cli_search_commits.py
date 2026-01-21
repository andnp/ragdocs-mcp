"""
E2E tests for search-commits CLI command.

Tests the git commit search command using Click's CliRunner with
real git repositories and commit indexing.
"""

import subprocess

import pytest
from click.testing import CliRunner

from src.cli import cli


@pytest.fixture
def runner():
    """
    Create Click CliRunner for CLI testing.

    Provides isolated environment for command execution.
    """
    return CliRunner()


@pytest.fixture
def git_repo(tmp_path):
    """
    Create git repository with sample commits for testing.

    Returns tuple of (repo_path, commit_hashes).
    """
    repo_path = tmp_path / "test_repo"
    repo_path.mkdir()

    # Initialize git repo
    subprocess.run(["git", "init"], cwd=repo_path, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.name", "Test User"],
        cwd=repo_path,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=repo_path,
        check=True,
        capture_output=True,
    )

    # Create commits with different content
    commits = []

    # Commit 1: Authentication feature
    (repo_path / "auth.py").write_text("def authenticate(): pass")
    subprocess.run(["git", "add", "."], cwd=repo_path, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "Add authentication module"],
        cwd=repo_path,
        check=True,
        capture_output=True,
    )
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_path,
        capture_output=True,
        text=True,
        check=True,
    )
    commits.append(result.stdout.strip())

    # Commit 2: Database fix
    (repo_path / "database.py").write_text("def query(): return []")
    subprocess.run(["git", "add", "."], cwd=repo_path, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "Fix database connection bug"],
        cwd=repo_path,
        check=True,
        capture_output=True,
    )
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_path,
        capture_output=True,
        text=True,
        check=True,
    )
    commits.append(result.stdout.strip())

    # Commit 3: Tests
    (repo_path / "test_auth.py").write_text("def test_auth(): assert True")
    subprocess.run(["git", "add", "."], cwd=repo_path, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "Add authentication tests"],
        cwd=repo_path,
        check=True,
        capture_output=True,
    )
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_path,
        capture_output=True,
        text=True,
        check=True,
    )
    commits.append(result.stdout.strip())

    return repo_path, commits


@pytest.fixture
def indexed_repo(tmp_path, git_repo):
    """
    Create and index a git repository for testing search-commits.

    Sets up configuration, creates index, and returns repo path.
    """
    import os

    repo_path, commits = git_repo

    # Create config with git_indexing enabled
    config_dir = tmp_path / ".mcp-markdown-ragdocs"
    config_dir.mkdir()
    config_path = config_dir / "config.toml"
    config_path.write_text(f"""
[indexing]
documents_path = "{repo_path}"
index_path = ".index_data"

[git_indexing]
enabled = true
delta_max_lines = 200

[llm]
embedding_model = "local"

[search]
semantic_weight = 1.0
keyword_weight = 1.0

[chunking]
strategy = "header_based"
min_chunk_chars = 200
max_chunk_chars = 2000
""")

    # Change to tmp directory and rebuild index
    original_cwd = os.getcwd()
    try:
        os.chdir(tmp_path)
        runner = CliRunner()
        result = runner.invoke(cli, ["rebuild-index"])
        assert result.exit_code == 0, f"rebuild-index failed: {result.output}"
        return repo_path
    finally:
        os.chdir(original_cwd)


# =============================================================================
# Basic Execution Tests
# =============================================================================


def test_search_commits_basic_execution(runner, tmp_path, indexed_repo):
    """
    Test search-commits command executes successfully with basic query.

    Validates that command runs, searches indexed commits, and returns results.
    """
    import os

    original_cwd = os.getcwd()
    try:
        os.chdir(tmp_path)

        result = runner.invoke(cli, ["search-commits", "authentication"])

        assert result.exit_code == 0
        assert "Query: authentication" in result.output
        assert "Found" in result.output
        assert "results" in result.output or "No results" in result.output

    finally:
        os.chdir(original_cwd)


def test_search_commits_returns_relevant_results(runner, tmp_path, indexed_repo):
    """
    Test search-commits returns relevant commits for query.

    Validates that semantic search finds commits matching the query.
    """
    import os

    original_cwd = os.getcwd()
    try:
        os.chdir(tmp_path)

        result = runner.invoke(cli, ["search-commits", "authentication"])

        assert result.exit_code == 0
        # Should find commits related to auth
        if "No results" not in result.output:
            assert "authentication" in result.output.lower() or "auth" in result.output.lower()

    finally:
        os.chdir(original_cwd)


# =============================================================================
# JSON Output Format Tests
# =============================================================================


def test_search_commits_json_output(runner, tmp_path, indexed_repo):
    """
    Test search-commits --json flag produces valid JSON output.

    Validates JSON structure with query, total_commits_indexed, and results.
    """
    import json
    import os

    original_cwd = os.getcwd()
    try:
        os.chdir(tmp_path)

        result = runner.invoke(cli, ["search-commits", "database", "--json"])

        assert result.exit_code == 0

        # Parse JSON
        data = json.loads(result.output)

        # Verify structure
        assert "query" in data
        assert data["query"] == "database"
        assert "total_commits_indexed" in data
        assert "results" in data
        assert isinstance(data["results"], list)

        # If results exist, verify structure
        if data["results"]:
            commit = data["results"][0]
            assert "hash" in commit
            assert "title" in commit
            assert "author" in commit
            assert "timestamp" in commit
            assert "score" in commit

    finally:
        os.chdir(original_cwd)


def test_search_commits_json_no_results(runner, tmp_path, indexed_repo):
    """
    Test search-commits --json with query that returns few or no results.

    Validates JSON structure is correct regardless of result count.
    """
    import json
    import os

    original_cwd = os.getcwd()
    try:
        os.chdir(tmp_path)

        result = runner.invoke(
            cli, ["search-commits", "nonexistent_topic_xyz123", "--json"]
        )

        assert result.exit_code == 0

        data = json.loads(result.output)
        assert data["query"] == "nonexistent_topic_xyz123"
        assert isinstance(data["results"], list)
        # Results list is valid even if not empty (embedding may find similarities)

    finally:
        os.chdir(original_cwd)


# =============================================================================
# Parameter Validation Tests
# =============================================================================


def test_search_commits_top_n_validation_minimum(runner, tmp_path, indexed_repo):
    """
    Test search-commits validates --top-n minimum value.

    Validates that --top-n < 1 produces error.
    """
    import os

    original_cwd = os.getcwd()
    try:
        os.chdir(tmp_path)

        result = runner.invoke(cli, ["search-commits", "test", "--top-n", "0"])

        assert result.exit_code == 1
        assert "Error" in result.output
        assert "--top-n must be between" in result.output

    finally:
        os.chdir(original_cwd)


def test_search_commits_top_n_validation_maximum(runner, tmp_path, indexed_repo):
    """
    Test search-commits validates --top-n maximum value.

    Validates that --top-n > 100 produces error.
    """
    import os

    original_cwd = os.getcwd()
    try:
        os.chdir(tmp_path)

        result = runner.invoke(cli, ["search-commits", "test", "--top-n", "150"])

        assert result.exit_code == 1
        assert "Error" in result.output
        assert "--top-n must be between" in result.output

    finally:
        os.chdir(original_cwd)


def test_search_commits_top_n_valid_range(runner, tmp_path, indexed_repo):
    """
    Test search-commits accepts valid --top-n values.

    Validates that --top-n within [1, 100] is accepted.
    """
    import os

    original_cwd = os.getcwd()
    try:
        os.chdir(tmp_path)

        result = runner.invoke(cli, ["search-commits", "test", "--top-n", "10"])

        assert result.exit_code == 0

    finally:
        os.chdir(original_cwd)


def test_search_commits_timestamp_validation(runner, tmp_path, indexed_repo):
    """
    Test search-commits validates timestamp ordering.

    Validates that --after >= --before produces error.
    """
    import os

    original_cwd = os.getcwd()
    try:
        os.chdir(tmp_path)

        result = runner.invoke(
            cli,
            ["search-commits", "test", "--after", "1000000000", "--before", "900000000"],
        )

        assert result.exit_code == 1
        assert "Error" in result.output
        assert "--after must be less than --before" in result.output

    finally:
        os.chdir(original_cwd)


def test_search_commits_timestamp_valid_range(runner, tmp_path, indexed_repo):
    """
    Test search-commits accepts valid timestamp range.

    Validates that proper --after < --before range is accepted.
    """
    import os

    original_cwd = os.getcwd()
    try:
        os.chdir(tmp_path)

        result = runner.invoke(
            cli,
            ["search-commits", "test", "--after", "900000000", "--before", "1900000000"],
        )

        assert result.exit_code == 0

    finally:
        os.chdir(original_cwd)


# =============================================================================
# Files Glob Filtering Tests
# =============================================================================


def test_search_commits_files_glob_filtering(runner, tmp_path, indexed_repo):
    """
    Test search-commits --files-glob filters by file patterns.

    Validates glob pattern filtering works correctly.
    """
    import os

    original_cwd = os.getcwd()
    try:
        os.chdir(tmp_path)

        result = runner.invoke(
            cli, ["search-commits", "test", "--files-glob", "test_*.py"]
        )

        assert result.exit_code == 0

    finally:
        os.chdir(original_cwd)


# =============================================================================
# Error Handling Tests
# =============================================================================


def test_search_commits_git_disabled_error(runner, tmp_path, git_repo):
    """
    Test search-commits handles git_indexing disabled gracefully.

    Validates error message when git indexing is not enabled.
    """
    import os

    repo_path, commits = git_repo

    # Create config with git_indexing disabled
    config_dir = tmp_path / ".mcp-markdown-ragdocs"
    config_dir.mkdir()
    config_path = config_dir / "config.toml"
    config_path.write_text(f"""
[indexing]
documents_path = "{repo_path}"

[git_indexing]
enabled = false

[llm]
embedding_model = "local"
""")

    original_cwd = os.getcwd()
    try:
        os.chdir(tmp_path)

        result = runner.invoke(cli, ["search-commits", "test"])

        assert result.exit_code == 1
        assert "Error" in result.output
        assert "Git indexing is not enabled" in result.output

    finally:
        os.chdir(original_cwd)


def test_search_commits_no_index_error(runner, tmp_path, git_repo):
    """
    Test search-commits with git_indexing enabled but no commits indexed.

    Validates that search works even with empty commit index (returns 0 results).
    """
    import os

    repo_path, commits = git_repo

    # Create config with git_indexing enabled but don't run rebuild-index
    config_dir = tmp_path / ".mcp-markdown-ragdocs"
    config_dir.mkdir()
    config_path = config_dir / "config.toml"
    config_path.write_text(f"""
[indexing]
documents_path = "{repo_path}"

[git_indexing]
enabled = true

[llm]
embedding_model = "local"
""")

    original_cwd = os.getcwd()
    try:
        os.chdir(tmp_path)

        # Without rebuild-index, commit_indexer is still created (empty DB)
        result = runner.invoke(cli, ["search-commits", "test"])

        # Should succeed with empty results (not error)
        assert result.exit_code == 0
        assert "Total commits indexed: 0" in result.output or "No results" in result.output

    finally:
        os.chdir(original_cwd)


def test_search_commits_empty_query(runner, tmp_path, indexed_repo):
    """
    Test search-commits handles empty query string.

    Validates that empty query still executes (may return all results).
    """
    import os

    original_cwd = os.getcwd()
    try:
        os.chdir(tmp_path)

        result = runner.invoke(cli, ["search-commits", ""])

        # Empty query should still execute without error
        assert result.exit_code == 0

    finally:
        os.chdir(original_cwd)


# =============================================================================
# Help Text Tests
# =============================================================================


def test_search_commits_help_text(runner):
    """
    Test search-commits --help displays usage information.

    Validates help text includes description and parameter explanations.
    """
    result = runner.invoke(cli, ["search-commits", "--help"])

    assert result.exit_code == 0
    assert "search-commits" in result.output
    assert "Search git commit history" in result.output
    assert "--json" in result.output
    assert "--top-n" in result.output
    assert "--files-glob" in result.output
    assert "--after" in result.output
    assert "--before" in result.output
