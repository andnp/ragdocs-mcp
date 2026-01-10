import pytest

from src.search.path_utils import normalize_path, matches_any_excluded


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
