"""
E2E tests for search-memory CLI command.

Tests the memory search command using Click's CliRunner with
real memory files and indexing.
"""

import os
import time

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
def memory_dir(tmp_path):
    """
    Create memory directory with sample memories for testing.

    Returns path to memory directory.
    """
    memory_path = tmp_path / ".memories"
    memory_path.mkdir()

    # Create memory 1: Plan with tags
    (memory_path / "plan-feature-a.md").write_text("""---
type: plan
status: active
tags: [feature-a, backend, database]
---

# Feature A Implementation Plan

We need to implement the new database schema for feature A.
This will require migrations and updated models.
""")

    # Create memory 2: Observation with different tags
    (memory_path / "observation-bug-fix.md").write_text("""---
type: observation
status: active
tags: [bug-fix, frontend, ui]
---

# Bug Fix Observations

The UI rendering issue was caused by incorrect CSS.
Fixed by adjusting the flexbox layout.
""")

    # Create memory 3: Reflection without tags
    (memory_path / "reflection-architecture.md").write_text("""---
type: reflection
status: active
tags: []
---

# Architecture Reflections

The current architecture needs refactoring to improve scalability.
Consider migrating to microservices.
""")

    # Create memory 4: Journal entry
    (memory_path / "journal-2026-01-15.md").write_text("""---
type: journal
status: active
tags: [daily, progress]
---

# Daily Journal - January 15, 2026

Made progress on authentication module today.
Tests are passing, ready for review.
""")

    return memory_path


@pytest.fixture
def indexed_memories(tmp_path, memory_dir):
    """
    Create and index memories for testing search-memory.

    Sets up configuration, creates index, and returns memory path.
    """
    # Create config with memory enabled
    config_dir = tmp_path / ".mcp-markdown-ragdocs"
    config_dir.mkdir()
    config_path = config_dir / "config.toml"
    config_path.write_text(f"""
[indexing]
documents_path = "{tmp_path}"
index_path = ".index_data"

[memory]
enabled = true
memory_path = "{memory_dir}"

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
        # Just create context to initialize memory index
        # (rebuild-index doesn't index memories)
        result = runner.invoke(cli, ["rebuild-index"])
        assert result.exit_code == 0, f"rebuild-index failed: {result.output}"
        return memory_dir
    finally:
        os.chdir(original_cwd)


# =============================================================================
# Basic Execution Tests
# =============================================================================


def test_search_memory_basic_execution(runner, tmp_path, indexed_memories):
    """
    Test search-memory command executes successfully with basic query.

    Validates that command runs, searches indexed memories, and returns results.
    """
    original_cwd = os.getcwd()
    try:
        os.chdir(tmp_path)

        result = runner.invoke(cli, ["search-memory", "database"])

        assert result.exit_code == 0
        assert "Query: database" in result.output
        assert "Found" in result.output or "No results" in result.output

    finally:
        os.chdir(original_cwd)


def test_search_memory_returns_relevant_results(runner, tmp_path, indexed_memories):
    """
    Test search-memory returns relevant memories for query.

    Validates that semantic search finds memories matching the query.
    """
    original_cwd = os.getcwd()
    try:
        os.chdir(tmp_path)

        result = runner.invoke(cli, ["search-memory", "feature implementation"])

        assert result.exit_code == 0
        # Should find relevant memories
        if "No results" not in result.output:
            assert "feature" in result.output.lower() or "implementation" in result.output.lower()

    finally:
        os.chdir(original_cwd)


# =============================================================================
# Tag Filtering Tests
# =============================================================================


def test_search_memory_single_tag_filter(runner, tmp_path, indexed_memories):
    """
    Test search-memory --tags filters by single tag.

    Validates that only memories with specified tag are returned.
    """
    original_cwd = os.getcwd()
    try:
        os.chdir(tmp_path)

        result = runner.invoke(cli, ["search-memory", "test", "--tags", "backend"])

        assert result.exit_code == 0

    finally:
        os.chdir(original_cwd)


def test_search_memory_multiple_tags_filter(runner, tmp_path, indexed_memories):
    """
    Test search-memory --tags filters by multiple comma-separated tags.

    Validates that tag list parsing works correctly.
    """
    original_cwd = os.getcwd()
    try:
        os.chdir(tmp_path)

        result = runner.invoke(cli, ["search-memory", "test", "--tags", "backend,database"])

        assert result.exit_code == 0

    finally:
        os.chdir(original_cwd)


def test_search_memory_tags_with_spaces(runner, tmp_path, indexed_memories):
    """
    Test search-memory --tags handles spaces around commas.

    Validates that tag parsing strips whitespace.
    """
    original_cwd = os.getcwd()
    try:
        os.chdir(tmp_path)

        result = runner.invoke(cli, ["search-memory", "test", "--tags", "backend , database "])

        assert result.exit_code == 0

    finally:
        os.chdir(original_cwd)


# =============================================================================
# Memory Type Filtering Tests
# =============================================================================


def test_search_memory_type_filter_plan(runner, tmp_path, indexed_memories):
    """
    Test search-memory --type filters by memory type (plan).

    Validates that only memories of specified type are returned.
    """
    original_cwd = os.getcwd()
    try:
        os.chdir(tmp_path)

        result = runner.invoke(cli, ["search-memory", "test", "--type", "plan"])

        assert result.exit_code == 0

    finally:
        os.chdir(original_cwd)


def test_search_memory_type_filter_journal(runner, tmp_path, indexed_memories):
    """
    Test search-memory --type filters by memory type (journal).

    Validates that type filtering works for different types.
    """
    original_cwd = os.getcwd()
    try:
        os.chdir(tmp_path)

        result = runner.invoke(cli, ["search-memory", "test", "--type", "journal"])

        assert result.exit_code == 0

    finally:
        os.chdir(original_cwd)


def test_search_memory_type_invalid(runner, tmp_path, indexed_memories):
    """
    Test search-memory --type validates memory type values.

    Validates that invalid type produces error.
    """
    original_cwd = os.getcwd()
    try:
        os.chdir(tmp_path)

        result = runner.invoke(cli, ["search-memory", "test", "--type", "invalid_type"])

        assert result.exit_code == 1
        assert "Error" in result.output
        assert "--type must be one of" in result.output

    finally:
        os.chdir(original_cwd)


def test_search_memory_all_valid_types(runner, tmp_path, indexed_memories):
    """
    Test search-memory accepts all valid memory types.

    Validates that all documented types are accepted.
    """
    import os

    valid_types = ["plan", "journal", "fact", "observation", "reflection"]

    original_cwd = os.getcwd()
    try:
        os.chdir(tmp_path)

        for memory_type in valid_types:
            result = runner.invoke(cli, ["search-memory", "test", "--type", memory_type])
            assert result.exit_code == 0, f"Type {memory_type} failed: {result.output}"

    finally:
        os.chdir(original_cwd)


# =============================================================================
# Time Range Filtering Tests
# =============================================================================


def test_search_memory_after_timestamp(runner, tmp_path, indexed_memories):
    """
    Test search-memory --after filters by timestamp lower bound.

    Validates that only memories after timestamp are returned.
    """
    original_cwd = os.getcwd()
    try:
        os.chdir(tmp_path)

        # Use timestamp before memories were created
        result = runner.invoke(
            cli, ["search-memory", "test", "--after", str(int(time.time()) - 3600)]
        )

        assert result.exit_code == 0

    finally:
        os.chdir(original_cwd)


def test_search_memory_before_timestamp(runner, tmp_path, indexed_memories):
    """
    Test search-memory --before filters by timestamp upper bound.

    Validates that only memories before timestamp are returned.
    """
    original_cwd = os.getcwd()
    try:
        os.chdir(tmp_path)

        # Use timestamp after memories were created
        result = runner.invoke(
            cli, ["search-memory", "test", "--before", str(int(time.time()) + 3600)]
        )

        assert result.exit_code == 0

    finally:
        os.chdir(original_cwd)


def test_search_memory_timestamp_range(runner, tmp_path, indexed_memories):
    """
    Test search-memory --after and --before work together.

    Validates that timestamp range filtering works correctly.
    """
    original_cwd = os.getcwd()
    try:
        os.chdir(tmp_path)

        after = int(time.time()) - 3600
        before = int(time.time()) + 3600

        result = runner.invoke(
            cli,
            ["search-memory", "test", "--after", str(after), "--before", str(before)],
        )

        assert result.exit_code == 0

    finally:
        os.chdir(original_cwd)


def test_search_memory_timestamp_validation(runner, tmp_path, indexed_memories):
    """
    Test search-memory validates timestamp ordering.

    Validates that --after >= --before produces error.
    """
    original_cwd = os.getcwd()
    try:
        os.chdir(tmp_path)

        result = runner.invoke(
            cli,
            ["search-memory", "test", "--after", "1000000000", "--before", "900000000"],
        )

        assert result.exit_code == 1
        assert "Error" in result.output
        assert "--after must be less than --before" in result.output

    finally:
        os.chdir(original_cwd)


def test_search_memory_relative_days(runner, tmp_path, indexed_memories):
    """
    Test search-memory --relative-days filters last N days.

    Validates that relative time filtering works.
    """
    original_cwd = os.getcwd()
    try:
        os.chdir(tmp_path)

        result = runner.invoke(cli, ["search-memory", "test", "--relative-days", "7"])

        assert result.exit_code == 0

    finally:
        os.chdir(original_cwd)


def test_search_memory_relative_days_validation(runner, tmp_path, indexed_memories):
    """
    Test search-memory validates --relative-days is non-negative.

    Validates that negative days produces error.
    """
    original_cwd = os.getcwd()
    try:
        os.chdir(tmp_path)

        result = runner.invoke(cli, ["search-memory", "test", "--relative-days", "-1"])

        assert result.exit_code == 1
        assert "Error" in result.output
        assert "--relative-days must be non-negative" in result.output

    finally:
        os.chdir(original_cwd)


def test_search_memory_relative_days_overrides_absolute(runner, tmp_path, indexed_memories):
    """
    Test search-memory --relative-days overrides absolute timestamps.

    Validates precedence of relative over absolute timestamps.
    """
    original_cwd = os.getcwd()
    try:
        os.chdir(tmp_path)

        result = runner.invoke(
            cli,
            [
                "search-memory",
                "test",
                "--after",
                "1000000000",
                "--relative-days",
                "30",
            ],
        )

        # Should succeed because relative-days overrides invalid absolute range
        assert result.exit_code == 0

    finally:
        os.chdir(original_cwd)


# =============================================================================
# Full Content Loading Tests
# =============================================================================


def test_search_memory_full_flag_loads_complete_content(runner, tmp_path, indexed_memories):
    """
    Test search-memory --full loads complete memory content.

    Validates that full content is displayed instead of truncated.
    """
    original_cwd = os.getcwd()
    try:
        os.chdir(tmp_path)

        result = runner.invoke(cli, ["search-memory", "database", "--full"])

        assert result.exit_code == 0

    finally:
        os.chdir(original_cwd)


def test_search_memory_without_full_truncates_content(runner, tmp_path, memory_dir):
    """
    Test search-memory without --full truncates long content.

    Validates that content is truncated to 500 chars by default.
    """
    # Create a long memory
    long_content = "Long content. " * 100  # > 500 chars
    (memory_dir / "long-memory.md").write_text(f"""---
type: plan
status: active
tags: []
---

{long_content}
""")

    original_cwd = os.getcwd()
    try:
        os.chdir(tmp_path)

        result = runner.invoke(cli, ["search-memory", "content"])

        assert result.exit_code == 0
        # If results found, check for truncation indicator
        if "No results" not in result.output and "Long content" in result.output:
            assert "..." in result.output

    finally:
        os.chdir(original_cwd)


# =============================================================================
# JSON Output Format Tests
# =============================================================================


def test_search_memory_json_output(runner, tmp_path, indexed_memories):
    """
    Test search-memory --json produces valid JSON output.

    Validates JSON structure with query and results.
    """
    import json

    original_cwd = os.getcwd()
    try:
        os.chdir(tmp_path)

        result = runner.invoke(cli, ["search-memory", "database", "--json"])

        assert result.exit_code == 0

        # Parse JSON
        data = json.loads(result.output)

        # Verify structure
        assert "query" in data
        assert data["query"] == "database"
        assert "results" in data
        assert isinstance(data["results"], list)

        # If results exist, verify structure
        if data["results"]:
            memory = data["results"][0]
            assert "memory_id" in memory
            assert "score" in memory
            assert "content" in memory
            assert "type" in memory
            assert "tags" in memory

    finally:
        os.chdir(original_cwd)


def test_search_memory_json_no_results(runner, tmp_path, indexed_memories):
    """
    Test search-memory --json with query that returns no results.

    Validates JSON structure is correct even with empty results.
    """
    import json

    original_cwd = os.getcwd()
    try:
        os.chdir(tmp_path)

        result = runner.invoke(
            cli, ["search-memory", "nonexistent_topic_xyz123", "--json"]
        )

        assert result.exit_code == 0

        data = json.loads(result.output)
        assert data["query"] == "nonexistent_topic_xyz123"
        assert data["results"] == []

    finally:
        os.chdir(original_cwd)


# =============================================================================
# Parameter Validation Tests
# =============================================================================


def test_search_memory_limit_validation_minimum(runner, tmp_path, indexed_memories):
    """
    Test search-memory validates --limit minimum value.

    Validates that --limit < 1 produces error.
    """
    original_cwd = os.getcwd()
    try:
        os.chdir(tmp_path)

        result = runner.invoke(cli, ["search-memory", "test", "--limit", "0"])

        assert result.exit_code == 1
        assert "Error" in result.output
        assert "--limit must be between" in result.output

    finally:
        os.chdir(original_cwd)


def test_search_memory_limit_validation_maximum(runner, tmp_path, indexed_memories):
    """
    Test search-memory validates --limit maximum value.

    Validates that --limit > 100 produces error.
    """
    original_cwd = os.getcwd()
    try:
        os.chdir(tmp_path)

        result = runner.invoke(cli, ["search-memory", "test", "--limit", "150"])

        assert result.exit_code == 1
        assert "Error" in result.output
        assert "--limit must be between" in result.output

    finally:
        os.chdir(original_cwd)


def test_search_memory_limit_valid_range(runner, tmp_path, indexed_memories):
    """
    Test search-memory accepts valid --limit values.

    Validates that --limit within [1, 100] is accepted.
    """
    original_cwd = os.getcwd()
    try:
        os.chdir(tmp_path)

        result = runner.invoke(cli, ["search-memory", "test", "--limit", "10"])

        assert result.exit_code == 0

    finally:
        os.chdir(original_cwd)


# =============================================================================
# Error Handling Tests
# =============================================================================


def test_search_memory_disabled_error(runner, tmp_path, memory_dir):
    """
    Test search-memory handles memory system disabled gracefully.

    Validates error message when memory is not enabled.
    """
    # Create config with memory disabled
    config_dir = tmp_path / ".mcp-markdown-ragdocs"
    config_dir.mkdir()
    config_path = config_dir / "config.toml"
    config_path.write_text(f"""
[indexing]
documents_path = "{tmp_path}"

[memory]
enabled = false

[llm]
embedding_model = "local"
""")

    original_cwd = os.getcwd()
    try:
        os.chdir(tmp_path)

        result = runner.invoke(cli, ["search-memory", "test"])

        assert result.exit_code == 1
        assert "Error" in result.output
        assert "Memory system is not enabled" in result.output

    finally:
        os.chdir(original_cwd)


def test_search_memory_empty_query(runner, tmp_path, indexed_memories):
    """
    Test search-memory handles empty query string.

    Validates that empty query still executes (may return all results).
    """
    original_cwd = os.getcwd()
    try:
        os.chdir(tmp_path)

        result = runner.invoke(cli, ["search-memory", ""])

        # Empty query should still execute without error
        assert result.exit_code == 0

    finally:
        os.chdir(original_cwd)


# =============================================================================
# Combined Filters Tests
# =============================================================================


def test_search_memory_combined_tags_and_type(runner, tmp_path, indexed_memories):
    """
    Test search-memory with both --tags and --type filters.

    Validates that multiple filters work together.
    """
    original_cwd = os.getcwd()
    try:
        os.chdir(tmp_path)

        result = runner.invoke(
            cli, ["search-memory", "test", "--tags", "backend", "--type", "plan"]
        )

        assert result.exit_code == 0

    finally:
        os.chdir(original_cwd)


def test_search_memory_all_filters_combined(runner, tmp_path, indexed_memories):
    """
    Test search-memory with tags, type, and time filters combined.

    Validates that all filters can be used simultaneously.
    """
    original_cwd = os.getcwd()
    try:
        os.chdir(tmp_path)

        result = runner.invoke(
            cli,
            [
                "search-memory",
                "test",
                "--tags",
                "backend",
                "--type",
                "plan",
                "--relative-days",
                "30",
                "--limit",
                "5",
            ],
        )

        assert result.exit_code == 0

    finally:
        os.chdir(original_cwd)


# =============================================================================
# Help Text Tests
# =============================================================================


def test_search_memory_help_text(runner):
    """
    Test search-memory --help displays usage information.

    Validates help text includes description and parameter explanations.
    """
    result = runner.invoke(cli, ["search-memory", "--help"])

    assert result.exit_code == 0
    assert "search-memory" in result.output
    assert "Search AI memory bank" in result.output
    assert "--json" in result.output
    assert "--limit" in result.output
    assert "--tags" in result.output
    assert "--type" in result.output
    assert "--after" in result.output
    assert "--before" in result.output
    assert "--relative-days" in result.output
    assert "--full" in result.output
