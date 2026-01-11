"""
Unit tests for git repository discovery with nested repos.

GAP #3: Git discovery with nested repos (Medium/Low, Score 3.33)
"""

import subprocess
from pathlib import Path

import pytest

from src.git.repository import discover_git_repositories, is_git_available


def _init_git_repo(path: Path):
    """Initialize a git repository."""
    subprocess.run(["git", "init"], cwd=path, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.name", "Test User"],
        cwd=path, check=True, capture_output=True
    )
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=path, check=True, capture_output=True
    )


@pytest.mark.skipif(not is_git_available(), reason="git not available")
def test_discover_nested_git_repositories(tmp_path):
    """
    Discover git repositories stops at first .git directory.

    Tests that discovery finds root repo but doesn't descend into
    repos to find nested repos (current implementation behavior).
    Note: This documents current behavior where nested repos aren't found.
    """
    # Create root repo
    root_repo = tmp_path / "root_project"
    root_repo.mkdir()
    _init_git_repo(root_repo)

    # Create nested repo (e.g., git submodule pattern)
    nested_repo = root_repo / "lib" / "external"
    nested_repo.mkdir(parents=True)
    _init_git_repo(nested_repo)

    # Create another nested repo at same level
    sibling_repo = root_repo / "lib" / "another"
    sibling_repo.mkdir(parents=True)
    _init_git_repo(sibling_repo)

    repos = discover_git_repositories(
        documents_path=tmp_path,
        exclude_patterns=[],
        exclude_hidden_dirs=False
    )

    # Current implementation stops at first .git (doesn't descend)
    assert len(repos) == 1
    assert repos[0].parent == root_repo


@pytest.mark.skipif(not is_git_available(), reason="git not available")
def test_discover_git_stops_at_repo_boundary(tmp_path):
    """
    Discovery stops at repository boundary, doesn't descend into .git.

    Tests that .git directory contents are not traversed.
    """
    # Create repo with nested structure
    repo = tmp_path / "project"
    repo.mkdir()
    _init_git_repo(repo)

    # Create subdirectories that should NOT be traversed if inside .git
    git_internals = repo / ".git" / "objects"
    git_internals.mkdir(parents=True, exist_ok=True)

    # Create another repo inside .git (pathological case)
    # This should NOT be discovered as we stop at .git boundary
    weird_nested = repo / ".git" / "nested_repo"
    weird_nested.mkdir(parents=True)
    # Don't init - just testing we don't descend

    repos = discover_git_repositories(
        documents_path=tmp_path,
        exclude_patterns=[],
        exclude_hidden_dirs=False
    )

    # Should only find the main repo, not traverse .git contents
    assert len(repos) == 1
    assert repos[0].parent == repo


@pytest.mark.skipif(not is_git_available(), reason="git not available")
def test_discover_git_with_exclude_patterns(tmp_path):
    """
    Discovery stops at first repo, nested repos not discovered.

    Tests that excluded paths work but current implementation
    stops descending once a .git is found.
    """
    # Create multiple repos
    main_repo = tmp_path / "main"
    main_repo.mkdir()
    _init_git_repo(main_repo)

    # Create repo in vendor directory (would be excluded if found)
    vendor_repo = tmp_path / "main" / "vendor" / "lib"
    vendor_repo.mkdir(parents=True)
    _init_git_repo(vendor_repo)

    # Create repo in node_modules (would be excluded if found)
    node_repo = tmp_path / "main" / "node_modules" / "package"
    node_repo.mkdir(parents=True)
    _init_git_repo(node_repo)

    # Create valid nested repo
    valid_nested = tmp_path / "main" / "subproject"
    valid_nested.mkdir()
    _init_git_repo(valid_nested)

    repos = discover_git_repositories(
        documents_path=tmp_path,
        exclude_patterns=["**/vendor/**", "**/node_modules/**"],
        exclude_hidden_dirs=False
    )

    # Current behavior: stops at main repo
    assert len(repos) == 1
    assert repos[0].parent == main_repo


@pytest.mark.skipif(not is_git_available(), reason="git not available")
def test_discover_git_with_hidden_dir_exclusion(tmp_path):
    """
    Discovery excludes hidden directories except .git itself.

    Tests that hidden dirs are skipped but .git is still found.
    """
    # Create normal repo
    normal_repo = tmp_path / "normal"
    normal_repo.mkdir()
    _init_git_repo(normal_repo)

    # Create repo in hidden directory
    hidden_dir = tmp_path / ".hidden_project"
    hidden_dir.mkdir()
    _init_git_repo(hidden_dir)

    # Create repo inside a hidden subdirectory
    nested_hidden = tmp_path / "normal" / ".cache" / "repo"
    nested_hidden.mkdir(parents=True)
    _init_git_repo(nested_hidden)

    repos = discover_git_repositories(
        documents_path=tmp_path,
        exclude_patterns=[],
        exclude_hidden_dirs=True  # Default behavior
    )

    # Should find normal repo, but not hidden ones
    assert len(repos) == 1
    assert repos[0].parent == normal_repo


@pytest.mark.skipif(not is_git_available(), reason="git not available")
def test_discover_git_empty_directory_tree(tmp_path):
    """
    Discovery handles empty directory tree gracefully.

    Tests that no repos found returns empty list.
    """
    # Create directories but no git repos
    (tmp_path / "dir1").mkdir()
    (tmp_path / "dir2" / "subdir").mkdir(parents=True)
    (tmp_path / "dir3").mkdir()

    repos = discover_git_repositories(
        documents_path=tmp_path,
        exclude_patterns=[],
        exclude_hidden_dirs=True
    )

    assert repos == []


@pytest.mark.skipif(not is_git_available(), reason="git not available")
def test_discover_git_deeply_nested_repositories(tmp_path):
    """
    Discovery stops at first .git in nested chain.

    Tests that discovery handles deep nesting but stops at first repo.
    """
    # Create deeply nested structure
    current = tmp_path
    depth = 10

    for i in range(depth):
        current = current / f"level_{i}"
        current.mkdir()

        # Create repo at some levels
        if i % 3 == 0:
            _init_git_repo(current)

    repos = discover_git_repositories(
        documents_path=tmp_path,
        exclude_patterns=[],
        exclude_hidden_dirs=False
    )

    # Current behavior: finds first repo (level_0), stops there
    assert len(repos) == 1
    assert "level_0" in str(repos[0])


@pytest.mark.skipif(not is_git_available(), reason="git not available")
def test_discover_git_with_symlinks_to_repos(tmp_path):
    """
    Discovery handles symlinks to git repositories.

    Tests that symlinks don't cause infinite loops.
    Note: os.walk with followlinks=False (default) doesn't follow symlinks.
    """
    # Create actual repo
    real_repo = tmp_path / "real_repo"
    real_repo.mkdir()
    _init_git_repo(real_repo)

    # Create directory that will have symlink
    link_container = tmp_path / "links"
    link_container.mkdir()

    # Create symlink to repo
    symlink_path = link_container / "linked_repo"
    symlink_path.symlink_to(real_repo)

    repos = discover_git_repositories(
        documents_path=tmp_path,
        exclude_patterns=[],
        exclude_hidden_dirs=False
    )

    # Should find the real repo
    # Symlink behavior depends on os.walk settings
    # By default, os.walk doesn't follow symlinks
    assert len(repos) >= 1
    assert real_repo in [r.parent for r in repos]


@pytest.mark.skipif(not is_git_available(), reason="git not available")
def test_discover_git_concurrent_nested_repos(tmp_path):
    """
    Discovery stops at root repo, doesn't find nested siblings.

    Tests that sibling nested repos aren't found with current behavior.
    """
    root = tmp_path / "project"
    root.mkdir()
    _init_git_repo(root)

    # Create multiple nested repos in different directories
    modules = root / "modules"
    modules.mkdir()

    module_a = modules / "module_a"
    module_a.mkdir()
    _init_git_repo(module_a)

    module_b = modules / "module_b"
    module_b.mkdir()
    _init_git_repo(module_b)

    module_c = modules / "module_c"
    module_c.mkdir()
    _init_git_repo(module_c)

    repos = discover_git_repositories(
        documents_path=tmp_path,
        exclude_patterns=[],
        exclude_hidden_dirs=False
    )

    # Current behavior: stops at root repo
    assert len(repos) == 1
    assert repos[0].parent == root
