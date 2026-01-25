import pytest
from pathlib import Path

from src.search.path_utils import (
    normalize_path,
    matches_any_excluded,
    compute_doc_id,
    extract_doc_id_from_chunk_id,
    resolve_doc_path,
)


@pytest.fixture
def docs_root(tmp_path):
    return tmp_path / "docs"


def test_normalize_path_filename_only(docs_root):
    result = normalize_path("README.md", docs_root)
    assert result == "README"


def test_normalize_path_relative_path(docs_root):
    result = normalize_path("docs/api.md", docs_root)
    assert result == "docs/api"


def test_normalize_path_absolute_path(docs_root):
    abs_path = str(docs_root / "docs" / "api.md")
    result = normalize_path(abs_path, docs_root)
    assert result == "docs/api"


def test_normalize_path_no_extension(docs_root):
    result = normalize_path("README", docs_root)
    assert result == "README"


def test_normalize_path_multiple_dots(docs_root):
    result = normalize_path("file.test.md", docs_root)
    assert result == "file.test"


def test_normalize_path_nested_directory(docs_root):
    result = normalize_path("a/b/c/file.md", docs_root)
    assert result == "a/b/c/file"


def test_normalize_path_outside_docs_root(docs_root):
    outside_path = "/other/location/file.md"
    result = normalize_path(outside_path, docs_root)
    assert result == "/other/location/file"


def test_normalize_path_markdown_extension(docs_root):
    result = normalize_path("guide.markdown", docs_root)
    assert result == "guide"


def test_matches_any_excluded_exact_match(docs_root):
    excluded = {"docs/api"}
    result = matches_any_excluded("docs/api.md", excluded, docs_root)
    assert result is True


def test_matches_any_excluded_filename_match(docs_root):
    excluded = {"README"}
    result = matches_any_excluded("docs/README.md", excluded, docs_root)
    assert result is True


def test_matches_any_excluded_no_match(docs_root):
    excluded = {"other", "docs/guide"}
    result = matches_any_excluded("docs/api.md", excluded, docs_root)
    assert result is False


def test_matches_any_excluded_absolute_path(docs_root):
    excluded = {"docs/api"}
    abs_path = str(docs_root / "docs" / "api.md")
    result = matches_any_excluded(abs_path, excluded, docs_root)
    assert result is True


def test_matches_any_excluded_case_sensitive(docs_root):
    excluded = {"readme"}
    result = matches_any_excluded("README.md", excluded, docs_root)
    assert result is False


def test_matches_any_excluded_empty_set(docs_root):
    excluded = set()
    result = matches_any_excluded("docs/api.md", excluded, docs_root)
    assert result is False


def test_matches_any_excluded_multiple_exclusions(docs_root):
    excluded = {"README", "api", "docs/guide"}
    assert matches_any_excluded("README.md", excluded, docs_root) is True
    assert matches_any_excluded("docs/api.md", excluded, docs_root) is True
    assert matches_any_excluded("docs/guide.md", excluded, docs_root) is True
    assert matches_any_excluded("other.md", excluded, docs_root) is False


def test_normalize_path_with_spaces(docs_root):
    result = normalize_path("my file.md", docs_root)
    assert result == "my file"


def test_matches_any_excluded_with_spaces(docs_root):
    excluded = {"my file"}
    result = matches_any_excluded("my file.md", excluded, docs_root)
    assert result is True


# Tests for compute_doc_id


def test_compute_doc_id_normal():
    """Test standard doc_id computation."""
    file_path = Path("/docs/guide/setup.md")
    docs_root = Path("/docs")
    assert compute_doc_id(file_path, docs_root) == "guide/setup"


def test_compute_doc_id_nested():
    """Test deeply nested path."""
    file_path = Path("/docs/api/v2/auth/jwt.md")
    docs_root = Path("/docs")
    assert compute_doc_id(file_path, docs_root) == "api/v2/auth/jwt"


def test_compute_doc_id_outside_root():
    """Test file outside docs root (fallback to absolute)."""
    file_path = Path("/tmp/external.md")
    docs_root = Path("/docs")
    result = compute_doc_id(file_path, docs_root)
    assert result == "/tmp/external"  # Absolute path, no extension


def test_compute_doc_id_root_file():
    """Test file at docs root."""
    file_path = Path("/docs/README.md")
    docs_root = Path("/docs")
    assert compute_doc_id(file_path, docs_root) == "README"


def test_compute_doc_id_backslash_normalization():
    """Test that backslashes are converted to forward slashes."""
    file_path = Path("/docs/dir/subdir/file.md")
    docs_root = Path("/docs")
    result = compute_doc_id(file_path, docs_root)
    # Should always use forward slashes regardless of platform
    assert "/" in result
    assert "\\" not in result


# Tests for extract_doc_id_from_chunk_id


def test_extract_doc_id_hash_separator():
    """Test chunk_id with hash separator (preferred format)."""
    assert extract_doc_id_from_chunk_id("guide/setup#chunk_0") == "guide/setup"


def test_extract_doc_id_hash_separator_nested():
    """Test chunk_id with nested path and hash separator."""
    assert extract_doc_id_from_chunk_id("api/v2/auth/jwt#chunk_5") == "api/v2/auth/jwt"


def test_extract_doc_id_underscore_separator():
    """Test chunk_id with underscore separator (legacy)."""
    assert extract_doc_id_from_chunk_id("guide_setup_chunk_0") == "guide_setup"


def test_extract_doc_id_underscore_separator_multiple():
    """Test chunk_id with multiple underscores before chunk suffix."""
    assert extract_doc_id_from_chunk_id("my_doc_name_chunk_3") == "my_doc_name"


def test_extract_doc_id_no_separator():
    """Test chunk_id without valid separator."""
    assert extract_doc_id_from_chunk_id("just_a_doc") == "just_a_doc"


def test_extract_doc_id_hash_priority():
    """Test that hash separator takes priority over underscore."""
    # If both separators present, hash should win
    assert extract_doc_id_from_chunk_id("doc_name#chunk_0") == "doc_name"


def test_extract_doc_id_only_chunk_suffix():
    """Test chunk_id that is only the chunk suffix (returns as-is with warning)."""
    # When input doesn't match expected format, return as-is
    assert extract_doc_id_from_chunk_id("chunk_0") == "chunk_0"


# Tests for resolve_doc_path


def test_resolve_doc_path_found(tmp_path):
    """Test resolving doc_id to existing file."""
    docs_root = tmp_path / "docs"
    docs_root.mkdir()
    test_file = docs_root / "guide" / "setup.md"
    test_file.parent.mkdir(parents=True)
    test_file.write_text("# Setup")

    result = resolve_doc_path("guide/setup", docs_root)
    assert result == test_file.resolve()


def test_resolve_doc_path_not_found(tmp_path):
    """Test resolving non-existent doc_id."""
    docs_root = tmp_path / "docs"
    docs_root.mkdir()

    result = resolve_doc_path("missing/doc", docs_root)
    assert result is None


def test_resolve_doc_path_multiple_extensions(tmp_path):
    """Test trying multiple extensions."""
    docs_root = tmp_path / "docs"
    docs_root.mkdir()
    test_file = docs_root / "readme.txt"
    test_file.write_text("README")

    result = resolve_doc_path("readme", docs_root, extensions=[".md", ".txt"])
    assert result == test_file.resolve()


def test_resolve_doc_path_prefers_first_extension(tmp_path):
    """Test that first matching extension is returned."""
    docs_root = tmp_path / "docs"
    docs_root.mkdir()
    md_file = docs_root / "readme.md"
    txt_file = docs_root / "readme.txt"
    md_file.write_text("# README")
    txt_file.write_text("README")

    # Should return .md since it's first in the list
    result = resolve_doc_path("readme", docs_root, extensions=[".md", ".txt"])
    assert result == md_file.resolve()


def test_resolve_doc_path_nested(tmp_path):
    """Test resolving nested doc_id."""
    docs_root = tmp_path / "docs"
    docs_root.mkdir()
    nested_file = docs_root / "api" / "v2" / "auth.md"
    nested_file.parent.mkdir(parents=True)
    nested_file.write_text("# Auth")

    result = resolve_doc_path("api/v2/auth", docs_root)
    assert result == nested_file.resolve()


def test_resolve_doc_path_default_extensions(tmp_path):
    """Test that default extensions are [.md, .txt]."""
    docs_root = tmp_path / "docs"
    docs_root.mkdir()
    md_file = docs_root / "doc.md"
    md_file.write_text("content")

    # Should work without specifying extensions
    result = resolve_doc_path("doc", docs_root)
    assert result == md_file.resolve()


def test_resolve_doc_path_backslash_in_doc_id(tmp_path):
    """Test that doc_id with backslashes is normalized."""
    docs_root = tmp_path / "docs"
    docs_root.mkdir()
    test_file = docs_root / "dir" / "file.md"
    test_file.parent.mkdir(parents=True)
    test_file.write_text("content")

    # Should handle backslashes (Windows-style paths)
    result = resolve_doc_path("dir\\file", docs_root)
    assert result == test_file.resolve()


def test_resolve_doc_path_only_returns_files(tmp_path):
    """Test that directories are not returned as valid paths."""
    docs_root = tmp_path / "docs"
    docs_root.mkdir()
    dir_path = docs_root / "subdir.md"  # Directory with .md extension
    dir_path.mkdir()

    result = resolve_doc_path("subdir", docs_root)
    assert result is None  # Should not match directory
