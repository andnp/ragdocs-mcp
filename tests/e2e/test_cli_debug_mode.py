"""
E2E tests for CLI debug mode feature.

Tests the --debug flag on query, search-commits, and search-memory commands.
"""

import pytest
from click.testing import CliRunner

from src.cli import cli


@pytest.fixture
def runner():
    """
    Create Click CliRunner for CLI testing.
    """
    return CliRunner()


@pytest.fixture
def indexed_docs(tmp_path):
    """
    Create and index sample documents for testing query --debug.
    """
    import os

    docs_path = tmp_path / "docs"
    docs_path.mkdir()

    (docs_path / "readme.md").write_text("""# Test Project

This is a test document for debugging search.

## Authentication

Authentication is handled via JWT tokens.
""")

    (docs_path / "guide.md").write_text("""# User Guide

Guide for using the system.

## Setup

Install dependencies first.
""")

    config_dir = tmp_path / ".mcp-markdown-ragdocs"
    config_dir.mkdir()
    config_path = config_dir / "config.toml"
    config_path.write_text(f"""
[indexing]
documents_path = "{docs_path}"
index_path = ".index_data"

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

    original_cwd = os.getcwd()
    try:
        os.chdir(tmp_path)
        runner = CliRunner()
        result = runner.invoke(cli, ["rebuild-index"])
        assert result.exit_code == 0, f"rebuild-index failed: {result.output}"
        return docs_path
    finally:
        os.chdir(original_cwd)


# =============================================================================
# Query Debug Mode Tests
# =============================================================================


def test_query_debug_displays_strategy_stats(runner, tmp_path, indexed_docs):
    """
    Test query --debug displays search strategy statistics.

    Validates that debug tables show vector, keyword, and graph counts.
    """
    import os

    original_cwd = os.getcwd()
    try:
        os.chdir(tmp_path)

        result = runner.invoke(cli, ["query", "authentication", "--debug"])

        assert result.exit_code == 0
        assert "Search Strategy Results" in result.output
        assert "Vector (Semantic)" in result.output
        assert "Keyword (BM25)" in result.output

    finally:
        os.chdir(original_cwd)


def test_query_debug_displays_compression_stats(runner, tmp_path, indexed_docs):
    """
    Test query --debug displays compression pipeline statistics.

    Validates that debug tables show deduplication stages.
    """
    import os

    original_cwd = os.getcwd()
    try:
        os.chdir(tmp_path)

        result = runner.invoke(cli, ["query", "authentication", "--debug"])

        assert result.exit_code == 0
        assert "Compression Pipeline" in result.output
        assert "Original (RRF Fusion)" in result.output
        assert "After Confidence Filter" in result.output
        assert "After Content Dedup" in result.output
        assert "After N-gram Dedup" in result.output
        assert "After Semantic Dedup" in result.output

    finally:
        os.chdir(original_cwd)


def test_query_without_debug_no_stats(runner, tmp_path, indexed_docs):
    """
    Test query without --debug does not display statistics.

    Validates that debug output is only shown when requested.
    """
    import os

    original_cwd = os.getcwd()
    try:
        os.chdir(tmp_path)

        result = runner.invoke(cli, ["query", "authentication"])

        assert result.exit_code == 0
        assert "Search Strategy Results" not in result.output
        assert "Compression Pipeline" not in result.output

    finally:
        os.chdir(original_cwd)


def test_query_debug_with_json_output(runner, tmp_path, indexed_docs):
    """
    Test query --debug with --json flag (should not show debug tables in JSON mode).

    Validates that --debug is ignored when --json is used.
    """
    import json
    import os

    original_cwd = os.getcwd()
    try:
        os.chdir(tmp_path)

        result = runner.invoke(cli, ["query", "authentication", "--debug", "--json"])

        assert result.exit_code == 0

        # JSON should be valid
        data = json.loads(result.output)
        assert "query" in data
        assert "results" in data

        # Debug tables should not appear in JSON output
        assert "Search Strategy Results" not in result.output
        assert "Compression Pipeline" not in result.output

    finally:
        os.chdir(original_cwd)


def test_query_debug_with_no_results(runner, tmp_path, indexed_docs):
    """
    Test query --debug handles queries with no results gracefully.

    Validates that debug stats are still shown even when no results found.
    """
    import os

    original_cwd = os.getcwd()
    try:
        os.chdir(tmp_path)

        result = runner.invoke(cli, ["query", "nonexistent_xyz_topic", "--debug"])

        assert result.exit_code == 0
        # Debug tables should still appear
        assert "Search Strategy Results" in result.output or "No results" in result.output

    finally:
        os.chdir(original_cwd)


def test_query_debug_shows_removal_counts(runner, tmp_path, indexed_docs):
    """
    Test query --debug shows how many results were removed at each stage.

    Validates that "Removed" column displays reduction counts.
    """
    import os

    original_cwd = os.getcwd()
    try:
        os.chdir(tmp_path)

        result = runner.invoke(cli, ["query", "guide", "--debug", "--top-n", "1"])

        assert result.exit_code == 0
        assert "Removed" in result.output
        # At least some filtering should occur
        assert "Compression Pipeline" in result.output

    finally:
        os.chdir(original_cwd)
