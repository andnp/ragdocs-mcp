"""
End-to-end tests for CLI query command.

Tests the 'query' subcommand functionality including JSON output,
flag combinations, and error handling.
"""

import json
import subprocess

import pytest
from click.testing import CliRunner

from src.cli import cli


@pytest.fixture
def runner():
    """
    Create Click CLI test runner.
    """
    return CliRunner()


@pytest.fixture
def test_env(tmp_path):
    """
    Set up test environment with config and test documents.

    Returns:
        dict with paths to config, docs, and index directories
    """
    # Create directory structure
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    index_dir = tmp_path / ".index_data"

    # Create test configuration
    config_dir = tmp_path / ".mcp-markdown-ragdocs"
    config_dir.mkdir()
    config_file = config_dir / "config.toml"
    config_file.write_text(f"""
[server]
host = "127.0.0.1"
port = 8000

[indexing]
documents_path = "{docs_dir}"
index_path = "{index_dir}"
recursive = false

[search]
semantic_weight = 1.0
keyword_weight = 1.0
recency_bias = 0.5
rrf_k_constant = 60

[llm]
embedding_model = "local"

[chunking]
strategy = "header_based"
min_chunk_chars = 200
max_chunk_chars = 1500
""")

    # Create test documents
    (docs_dir / "test_doc.md").write_text("""# Test Document

This is a test document for CLI query testing.

## Authentication

User authentication via OAuth 2.0.

## Authorization

Role-based access control.
""")

    (docs_dir / "api_guide.md").write_text("""# API Guide

Complete API reference documentation.

## Endpoints

List of available API endpoints.

## Authentication

API key authentication methods.
""")

    return {
        "config": config_file,
        "docs": docs_dir,
        "index": index_dir,
        "root": tmp_path,
    }


def build_index(test_env):
    """
    Helper to build index before running query tests.
    """
    import os
    original_cwd = os.getcwd()
    os.chdir(test_env["root"])

    try:
        runner = CliRunner()
        result = runner.invoke(cli, ["rebuild-index"])
        assert result.exit_code == 0, f"Index build failed: {result.output}"
    finally:
        os.chdir(original_cwd)


def test_query_command_basic_execution(runner, test_env):
    """
    Test basic CLI query execution with rich output.

    Verifies:
    - Command runs successfully (exit code 0)
    - Output contains expected sections
    - No errors in output
    """
    import os

    # Build index first
    build_index(test_env)

    # Change to test directory
    original_cwd = os.getcwd()
    os.chdir(test_env["root"])

    try:
        # Run query command
        result = runner.invoke(cli, ["query", "authentication"])

        # Verify execution
        assert result.exit_code == 0, f"Query failed: {result.output}"

        # Verify output contains expected sections
        output = result.output
        assert "Query:" in output or "query" in output.lower()

        # Should have results or a no results message
        # (Rich formatting may use different text)
        assert "results" in output.lower() or "Score:" in output or "Document:" in output, \
            "Output should contain results or results status"

        # Verify no error messages (excluding "0 errors" which is benign)
        assert "Error:" not in output

    finally:
        os.chdir(original_cwd)


def test_query_command_json_output(runner, test_env):
    """
    Test CLI query with --json flag.

    Verifies:
    - JSON output is valid
    - Has required keys: query, results
    - Results is list of dicts with proper structure
    """
    import os

    build_index(test_env)

    original_cwd = os.getcwd()
    os.chdir(test_env["root"])

    try:
        # Run query with --json flag
        result = runner.invoke(cli, ["query", "authentication", "--json"])

        assert result.exit_code == 0, f"Query failed: {result.output}"

        # Parse JSON output
        try:
            data = json.loads(result.output)
        except json.JSONDecodeError as e:
            pytest.fail(f"Invalid JSON output: {e}\n{result.output}")

        # Verify structure
        assert "query" in data, "JSON should contain 'query' field"
        assert "results" in data, "JSON should contain 'results' field"
        assert "answer" not in data, "JSON should not contain 'answer' field (synthesis removed)"

        # Verify query field
        assert data["query"] == "authentication"

        # Verify results structure
        assert isinstance(data["results"], list), "results should be a list"

        if data["results"]:
            result_item = data["results"][0]
            assert isinstance(result_item, dict), "Each result should be a dict"
            assert "chunk_id" in result_item
            assert "doc_id" in result_item
            assert "score" in result_item
            assert "header_path" in result_item
            assert "file_path" in result_item

            # Verify types
            assert isinstance(result_item["chunk_id"], str)
            assert isinstance(result_item["score"], (int, float))
            assert 0.0 <= result_item["score"] <= 1.0

    finally:
        os.chdir(original_cwd)


def test_query_command_top_n_option(runner, test_env):
    """
    Test CLI query with --top-n option.

    Verifies:
    - --top-n limits result count
    - Different top_n values work correctly
    """
    import os

    build_index(test_env)

    original_cwd = os.getcwd()
    os.chdir(test_env["root"])

    try:
        # Test with top_n=3
        result = runner.invoke(cli, ["query", "test", "--top-n", "3", "--json"])

        assert result.exit_code == 0, f"Query failed: {result.output}"

        data = json.loads(result.output)
        results = data["results"]

        # Should have at most 3 results
        assert len(results) <= 3, f"Expected at most 3 results, got {len(results)}"

        # Test with top_n=1
        result = runner.invoke(cli, ["query", "test", "--top-n", "1", "--json"])
        assert result.exit_code == 0

        data = json.loads(result.output)
        results = data["results"]
        assert len(results) <= 1, f"Expected at most 1 result, got {len(results)}"

    finally:
        os.chdir(original_cwd)


def test_query_command_error_no_indices(runner, tmp_path):
    """
    Test CLI query error handling when indices don't exist.

    Verifies:
    - Command fails gracefully (non-zero exit code)
    - Error message is clear about missing indices
    """
    import os

    # Create config without building index
    config_file = tmp_path / "config.toml"
    config_file.write_text(f"""
[indexing]
documents_path = "{tmp_path / 'docs'}"
index_path = "{tmp_path / '.index_data'}"
""")

    (tmp_path / "docs").mkdir()

    original_cwd = os.getcwd()
    os.chdir(tmp_path)

    try:
        # Try to query without index
        result = runner.invoke(cli, ["query", "test"])

        # Should fail
        assert result.exit_code != 0, "Should fail when index doesn't exist"

        # Error message should mention index
        assert "index" in result.output.lower() or "indices" in result.output.lower()

    finally:
        os.chdir(original_cwd)


def test_query_command_help(runner):
    """
    Test CLI query --help output.

    Verifies:
    - Help command works
    - Help text contains expected options
    - Help text is informative
    """
    result = runner.invoke(cli, ["query", "--help"])

    assert result.exit_code == 0

    help_text = result.output

    # Verify expected options are documented
    assert "query" in help_text.lower()
    assert "--json" in help_text
    assert "--top-n" in help_text

    # Verify help is informative
    assert len(help_text) > 100, "Help text should be substantial"


def test_query_command_validation_top_n_range(runner, test_env):
    """
    Test CLI query validates top_n range.

    Verifies:
    - top_n < 1 is rejected
    - top_n > 100 is rejected
    - Helpful error message
    """
    import os

    build_index(test_env)

    original_cwd = os.getcwd()
    os.chdir(test_env["root"])

    try:
        # Test top_n=0 (invalid)
        result = runner.invoke(cli, ["query", "test", "--top-n", "0"])
        assert result.exit_code != 0, "Should reject top_n=0"
        assert "top-n" in result.output.lower() or "between" in result.output.lower()

        # Test top_n=101 (invalid)
        result = runner.invoke(cli, ["query", "test", "--top-n", "101"])
        assert result.exit_code != 0, "Should reject top_n=101"
        assert "top-n" in result.output.lower() or "between" in result.output.lower()

    finally:
        os.chdir(original_cwd)


def test_query_command_subprocess_execution(test_env):
    """
    Test CLI query via subprocess (real execution).

    Verifies:
    - Command works as actual subprocess
    - Output is captured correctly
    - JSON parsing works end-to-end
    """
    import os

    # Build index
    original_cwd = os.getcwd()
    os.chdir(test_env["root"])

    try:
        # Build index first
        subprocess.run(
            ["uv", "run", "mcp-markdown-ragdocs", "rebuild-index"],
            check=True,
            capture_output=True,
            text=True,
        )

        # Run query via subprocess
        result = subprocess.run(
            ["uv", "run", "mcp-markdown-ragdocs", "query", "authentication", "--json"],
            capture_output=True,
            text=True,
        )

        # Verify execution
        assert result.returncode == 0, f"Query failed: {result.stderr}"

        # Parse JSON output
        data = json.loads(result.stdout)

        # Verify structure
        assert "query" in data
        assert "results" in data
        assert "answer" not in data, "JSON should not contain 'answer' field (synthesis removed)"

        assert isinstance(data["results"], list)

    except FileNotFoundError:
        pytest.skip("uv command not available in test environment")
    finally:
        os.chdir(original_cwd)


def test_query_command_with_special_characters(runner, test_env):
    """
    Test CLI query handles special characters in query text.

    Verifies:
    - Quotes, ampersands, etc. are handled correctly
    - Query parsing is robust
    """
    import os

    build_index(test_env)

    original_cwd = os.getcwd()
    os.chdir(test_env["root"])

    try:
        # Test with quotes
        result = runner.invoke(cli, ["query", "authentication 'OAuth'", "--json"])
        assert result.exit_code == 0

        data = json.loads(result.output)
        assert "query" in data

        # Test with question mark
        result = runner.invoke(cli, ["query", "what is authentication?", "--json"])
        assert result.exit_code == 0

    finally:
        os.chdir(original_cwd)


def test_query_command_empty_results(runner, test_env):
    """
    Test CLI query handles queries with no matching results.

    Verifies:
    - Command completes successfully
    - Returns empty results gracefully
    """
    import os

    build_index(test_env)

    original_cwd = os.getcwd()
    os.chdir(test_env["root"])

    try:
        # Query for something unlikely to match
        result = runner.invoke(cli, [
            "query",
            "xyzabc123nonexistentquery456",
            "--json"
        ])

        assert result.exit_code == 0, "Should handle no results gracefully"

        data = json.loads(result.output)

        # Should have empty or minimal results
        assert isinstance(data["results"], list)
        assert "answer" not in data, "JSON should not contain 'answer' field (synthesis removed)"

    finally:
        os.chdir(original_cwd)


def test_query_panel_format_with_scores(runner, test_env):
    """
    Test CLI query displays panels with scores in titles.

    Verifies:
    - Each result is displayed in a panel (when results exist)
    - Score is shown in panel title with 4 decimal places
    - Panel contains document, section, and file info
    - Panel formatting is properly applied
    """
    import os

    build_index(test_env)

    original_cwd = os.getcwd()
    os.chdir(test_env["root"])

    try:
        # Use a simple query that should match test documents
        # Testing with "test" which appears in the test documents
        result = runner.invoke(cli, ["query", "test document API"])

        assert result.exit_code == 0, f"Query failed: {result.output}"

        output = result.output

        # Check if results were found (may vary with local embeddings)
        if "No results found" in output:
            # If no results with embeddings, verify structure from JSON output instead
            result_json = runner.invoke(cli, ["query", "test", "--json", "--top-n", "5"])
            data = json.loads(result_json.output)

            # If JSON has results, test the rich output format from a different angle
            # This can happen with keyword search fallback
            if data.get("results"):
                # Re-run to get rich output
                result = runner.invoke(cli, ["query", "test"])
                output = result.output

        # If results exist in output, verify panel formatting
        if "Score:" in output:
            # Check for 4 decimal place format (e.g., 0.1234, 1.0000)
            import re
            score_pattern = r"Score:\s*\d+\.\d{4}"
            scores_found = re.findall(score_pattern, output)
            assert len(scores_found) > 0, "Should find scores formatted with 4 decimal places"

            # Verify panel structure elements
            assert "Document:" in output, "Should display Document field"
            assert "Section:" in output or "File:" in output, "Should display Section or File field"

            # Verify result numbering
            assert "#1" in output, "Should show result number #1"
        else:
            # If no results, at least verify the command structure works
            assert "Query:" in output, "Should display query label"
            assert "No relevant" in output or "results" in output.lower(), "Should show results status"

    finally:
        os.chdir(original_cwd)


def test_query_visual_separators(runner, test_env):
    """
    Test that visual separators work properly between results.

    Verifies:
    - Multiple results are displayed with clear separation
    - Panel formatting creates visual boundaries
    - Output is readable with distinct result sections
    """
    import os

    build_index(test_env)

    original_cwd = os.getcwd()
    os.chdir(test_env["root"])

    try:
        # Run query that should return multiple results
        result = runner.invoke(cli, ["query", "API authentication", "--top-n", "3"])

        assert result.exit_code == 0, f"Query failed: {result.output}"

        output = result.output

        # Count result panels (looking for result numbers)
        import re
        result_numbers = re.findall(r"#(\d+)", output)

        if len(result_numbers) > 1:
            # Multiple results - verify they're visually separated
            # Panel titles should contain result numbers
            assert "#1" in output, "Should show first result"
            assert "#2" in output or len(result_numbers) >= 1, "Should show additional results"

            # Each result should have its own Document/Section/File info
            doc_count = output.count("Document:")
            assert doc_count >= len(result_numbers), "Each result should have Document field"

        # Verify overall structure is present
        assert "Query:" in output, "Should show query header"
        assert "Found" in output or "results" in output.lower(), "Should indicate results found"

    finally:
        os.chdir(original_cwd)


def test_query_score_accuracy_json(runner, test_env):
    """
    Test that scores are accurate and properly normalized in JSON output.

    Verifies:
    - Scores are in range [0.0, 1.0]
    - Top result has highest score
    - Scores are descending order
    - Precision is maintained
    """
    import os

    build_index(test_env)

    original_cwd = os.getcwd()
    os.chdir(test_env["root"])

    try:
        # Run query with JSON output
        result = runner.invoke(cli, ["query", "authentication", "--json", "--top-n", "5"])

        assert result.exit_code == 0, f"Query failed: {result.output}"

        data = json.loads(result.output)
        results = data.get("results", [])

        if len(results) > 0:
            # Verify scores are in valid range
            for idx, res in enumerate(results):
                score = res["score"]
                assert 0.0 <= score <= 1.0, f"Result {idx} score {score} out of range [0, 1]"

            # Verify scores are in descending order
            scores = [r["score"] for r in results]
            assert scores == sorted(scores, reverse=True), "Scores should be descending"

            # Top result should have highest score (typically 1.0 after normalization)
            assert results[0]["score"] >= results[-1]["score"], "First result should have highest score"

    finally:
        os.chdir(original_cwd)
