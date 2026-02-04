"""Unit tests for file discovery logic."""

import pytest
from pathlib import Path

from src.indexing.discovery import discover_files


@pytest.fixture
def docs_structure(tmp_path):
    """Create a realistic document structure for testing."""
    docs = tmp_path / "docs"
    docs.mkdir()

    # Top-level markdown
    (docs / "README.md").write_text("# README")
    (docs / "CONTRIBUTING.md").write_text("# Contributing")

    # Nested markdown
    api_dir = docs / "api"
    api_dir.mkdir()
    (api_dir / "overview.md").write_text("# API Overview")
    (api_dir / "endpoints.md").write_text("# Endpoints")

    # Plain text files
    (docs / "notes.txt").write_text("Plain text notes")

    # Hidden directory (should be excluded by default)
    hidden = docs / ".hidden"
    hidden.mkdir()
    (hidden / "secret.md").write_text("# Secret")

    return docs


class TestDiscoverFiles:
    """Tests for discover_files function."""

    def test_single_pattern_finds_matching_files(self, docs_structure):
        """Basic discovery with single glob pattern."""
        parsers = {"**/*.md": "markdown"}

        files = discover_files(docs_structure, parsers)

        assert len(files) >= 4
        assert any("README.md" in f for f in files)
        assert any("CONTRIBUTING.md" in f for f in files)
        assert any("overview.md" in f for f in files)
        assert any("endpoints.md" in f for f in files)

    def test_multiple_patterns_combine_results(self, docs_structure):
        """Discovery with multiple glob patterns."""
        parsers = {
            "**/*.md": "markdown",
            "**/*.txt": "plaintext",
        }

        files = discover_files(docs_structure, parsers)

        assert any("README.md" in f for f in files)
        assert any("notes.txt" in f for f in files)

    def test_non_matching_pattern_returns_empty(self, tmp_path):
        """Patterns with no matches return empty list."""
        docs = tmp_path / "empty"
        docs.mkdir()

        parsers = {"**/*.md": "markdown"}

        files = discover_files(docs, parsers)

        assert files == []

    def test_excludes_hidden_dirs_by_default(self, docs_structure):
        """Hidden directories are excluded by default."""
        parsers = {"**/*.md": "markdown"}

        files = discover_files(docs_structure, parsers)

        assert not any(".hidden" in f for f in files)

    def test_includes_hidden_dirs_when_disabled(self, docs_structure):
        """Hidden directories can be included with explicit pattern and exclude_hidden_dirs=False.
        
        Note: glob's ** pattern doesn't match hidden directories by default,
        so an explicit pattern must include them.
        """
        parsers = {
            "**/*.md": "markdown",
            ".hidden/*.md": "markdown",  # Explicit pattern for hidden dir
        }

        files = discover_files(
            docs_structure,
            parsers,
            exclude_hidden_dirs=False,
        )

        assert any(".hidden" in f for f in files)

    def test_exclude_patterns_filter_out_files(self, docs_structure):
        """Exclude patterns remove matching files from results."""
        parsers = {"**/*.md": "markdown"}

        files = discover_files(
            docs_structure,
            parsers,
            include_patterns=["*"],
            exclude_patterns=["**/api/*"],
        )

        assert any("README.md" in f for f in files)
        assert not any("overview.md" in f for f in files)
        assert not any("endpoints.md" in f for f in files)

    def test_returns_sorted_list(self, docs_structure):
        """Results are returned in sorted order."""
        parsers = {"**/*.md": "markdown"}

        files = discover_files(docs_structure, parsers)

        assert files == sorted(files)

    def test_returns_absolute_paths(self, docs_structure):
        """All returned paths are absolute."""
        parsers = {"**/*.md": "markdown"}

        files = discover_files(docs_structure, parsers)

        # glob.glob returns absolute paths when given absolute base
        for f in files:
            assert Path(f).is_absolute()

    def test_non_recursive_only_searches_top_level(self, docs_structure):
        """Non-recursive search only finds top-level files."""
        # Use pattern without ** prefix for non-recursive
        parsers = {"*.md": "markdown"}

        files = discover_files(docs_structure, parsers, recursive=False)

        assert any("README.md" in f for f in files)
        assert any("CONTRIBUTING.md" in f for f in files)
        # Nested files should not be found with non-** pattern
        assert not any("overview.md" in f for f in files)

    def test_accepts_path_object(self, docs_structure):
        """Documents path can be a Path object."""
        parsers = {"**/*.md": "markdown"}

        files = discover_files(Path(docs_structure), parsers)

        assert len(files) >= 2

    def test_accepts_string_path(self, docs_structure):
        """Documents path can be a string."""
        parsers = {"**/*.md": "markdown"}

        files = discover_files(str(docs_structure), parsers)

        assert len(files) >= 2
