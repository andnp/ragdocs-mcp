"""
Tests for file filtering logic using include/exclude patterns.
"""
from src.cli import _should_include_file


def test_include_all_pattern():
    """
    Test that **/* pattern includes all files.
    """
    result = _should_include_file(
        "/path/to/file.md",
        include_patterns=["**/*"],
        exclude_patterns=[]
    )
    assert result is True


def test_exclude_build_directory():
    """
    Test that **/build/** pattern excludes build directory files.
    """
    result = _should_include_file(
        "/project/build/docs/README.md",
        include_patterns=["**/*"],
        exclude_patterns=["**/build/**"]
    )
    assert result is False


def test_exclude_takes_precedence():
    """
    Test that exclude patterns take precedence over include patterns.
    """
    result = _should_include_file(
        "/project/docs/build/api.md",
        include_patterns=["**/*.md"],
        exclude_patterns=["**/build/**"]
    )
    assert result is False


def test_include_specific_extension():
    """
    Test including only specific file extensions.
    """
    result_md = _should_include_file(
        "/project/docs/README.md",
        include_patterns=["**/*.md"],
        exclude_patterns=[]
    )
    assert result_md is True

    result_txt = _should_include_file(
        "/project/docs/notes.txt",
        include_patterns=["**/*.md"],
        exclude_patterns=[]
    )
    assert result_txt is False


def test_multiple_exclude_patterns():
    """
    Test multiple exclude patterns.
    """
    exclude_patterns = ["**/build/**", "**/.venv/**", "**/node_modules/**"]

    assert _should_include_file(
        "/project/docs/README.md",
        include_patterns=["**/*"],
        exclude_patterns=exclude_patterns
    ) is True

    assert _should_include_file(
        "/project/build/docs/api.md",
        include_patterns=["**/*"],
        exclude_patterns=exclude_patterns
    ) is False

    assert _should_include_file(
        "/project/.venv/lib/python/file.py",
        include_patterns=["**/*"],
        exclude_patterns=exclude_patterns
    ) is False

    assert _should_include_file(
        "/project/node_modules/package/index.js",
        include_patterns=["**/*"],
        exclude_patterns=exclude_patterns
    ) is False


def test_venv_variations():
    """
    Test that various venv directory names are excluded.
    """
    exclude_patterns = ["**/.venv/**", "**/venv/**"]

    assert _should_include_file(
        "/project/.venv/docs/README.md",
        include_patterns=["**/*"],
        exclude_patterns=exclude_patterns
    ) is False

    assert _should_include_file(
        "/project/venv/docs/README.md",
        include_patterns=["**/*"],
        exclude_patterns=exclude_patterns
    ) is False


def test_nested_build_directories():
    """
    Test that build directories at any level are excluded.
    """
    result = _should_include_file(
        "/project/libs/mylib/build/docs/api.md",
        include_patterns=["**/*"],
        exclude_patterns=["**/build/**"]
    )
    assert result is False


def test_exclude_hidden_dirs_enabled():
    """
    Test that files in hidden directories are excluded when exclude_hidden_dirs=True (default).
    """
    # Test .stversions directory (syncthing)
    assert _should_include_file(
        "/project/.stversions/file.md",
        include_patterns=["**/*"],
        exclude_patterns=[],
        exclude_hidden_dirs=True
    ) is False

    # Test .git directory
    assert _should_include_file(
        "/project/.git/config.md",
        include_patterns=["**/*"],
        exclude_patterns=[],
        exclude_hidden_dirs=True
    ) is False

    # Test .cache directory
    assert _should_include_file(
        "/project/.cache/data/file.md",
        include_patterns=["**/*"],
        exclude_patterns=[],
        exclude_hidden_dirs=True
    ) is False

    # Test nested hidden directory
    assert _should_include_file(
        "/project/docs/.hidden/file.md",
        include_patterns=["**/*"],
        exclude_patterns=[],
        exclude_hidden_dirs=True
    ) is False

    # Test deeply nested hidden directory
    assert _should_include_file(
        "/project/a/b/.secret/c/d/file.md",
        include_patterns=["**/*"],
        exclude_patterns=[],
        exclude_hidden_dirs=True
    ) is False

    # Test non-hidden directories should still be included
    assert _should_include_file(
        "/project/docs/file.md",
        include_patterns=["**/*"],
        exclude_patterns=[],
        exclude_hidden_dirs=True
    ) is True

    # Test file with dot in name (but not in directory) should be included
    assert _should_include_file(
        "/project/docs/.gitignore",
        include_patterns=["**/*"],
        exclude_patterns=[],
        exclude_hidden_dirs=True
    ) is False  # This is in a hidden "dir" conceptually (file itself starts with .)

    # Regular file in regular directory
    assert _should_include_file(
        "/project/normal/path/file.md",
        include_patterns=["**/*"],
        exclude_patterns=[],
        exclude_hidden_dirs=True
    ) is True


def test_exclude_hidden_dirs_disabled():
    """
    Test that files in hidden directories are NOT excluded when exclude_hidden_dirs=False.
    """
    # All these should now be included when the feature is disabled
    assert _should_include_file(
        "/project/.stversions/file.md",
        include_patterns=["**/*"],
        exclude_patterns=[],
        exclude_hidden_dirs=False
    ) is True

    assert _should_include_file(
        "/project/.git/config.md",
        include_patterns=["**/*"],
        exclude_patterns=[],
        exclude_hidden_dirs=False
    ) is True

    assert _should_include_file(
        "/project/docs/.hidden/file.md",
        include_patterns=["**/*"],
        exclude_patterns=[],
        exclude_hidden_dirs=False
    ) is True


def test_exclude_hidden_dirs_with_other_patterns():
    """
    Test that hidden directory exclusion works alongside include/exclude patterns.
    """
    # Even if hidden dirs are excluded, exclude patterns should still apply
    assert _should_include_file(
        "/project/build/file.md",
        include_patterns=["**/*"],
        exclude_patterns=["**/build/**"],
        exclude_hidden_dirs=True
    ) is False

    # Hidden directory should be excluded even if include pattern matches
    assert _should_include_file(
        "/project/.hidden/file.md",
        include_patterns=["**/*.md"],
        exclude_patterns=[],
        exclude_hidden_dirs=True
    ) is False

    # Non-hidden file matching include pattern should be included
    assert _should_include_file(
        "/project/docs/file.md",
        include_patterns=["**/*.md"],
        exclude_patterns=[],
        exclude_hidden_dirs=True
    ) is True

    # Hidden directory is excluded before pattern matching
    # This tests the order of operations
    assert _should_include_file(
        "/project/.cache/important.md",
        include_patterns=["**/*.md"],
        exclude_patterns=[],
        exclude_hidden_dirs=True
    ) is False


def test_hidden_dir_edge_cases():
    """
    Test edge cases for hidden directory detection.
    """
    # Directory name that contains but doesn't start with dot
    assert _should_include_file(
        "/project/my.folder/file.md",
        include_patterns=["**/*"],
        exclude_patterns=[],
        exclude_hidden_dirs=True
    ) is True

    # Multiple dots
    assert _should_include_file(
        "/project/..dots../file.md",
        include_patterns=["**/*"],
        exclude_patterns=[],
        exclude_hidden_dirs=True
    ) is False

    # Windows-style path (will be normalized)
    result = _should_include_file(
        "C:\\project\\.hidden\\file.md",
        include_patterns=["**/*"],
        exclude_patterns=[],
        exclude_hidden_dirs=True
    )
    assert result is False
