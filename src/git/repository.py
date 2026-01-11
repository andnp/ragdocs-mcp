"""Git repository discovery and commit listing."""

import logging
import os
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)


def discover_git_repositories(
    documents_path: Path,
    exclude_patterns: list[str],
    exclude_hidden_dirs: bool = True,
) -> list[Path]:
    """
    Recursively discover .git directories.

    Args:
        documents_path: Root path to search
        exclude_patterns: Glob patterns to exclude (e.g., '**/.venv/**')
        exclude_hidden_dirs: Skip hidden directories except .git

    Returns:
        List of absolute paths to .git directories
    """
    git_repos: list[Path] = []

    for root, dirs, _ in os.walk(documents_path, topdown=True):
        root_path = Path(root)

        # Check if current directory has .git
        git_dir = root_path / ".git"
        if git_dir.is_dir():
            # Check exclusion patterns
            should_exclude = False
            for pattern in exclude_patterns:
                if git_dir.match(pattern):
                    should_exclude = True
                    break

            if not should_exclude:
                git_repos.append(git_dir.resolve())
                logger.debug(f"Found git repository: {git_dir}")

            # Don't descend into this directory further
            dirs.clear()
            continue

        # Filter directories for descent
        dirs_to_remove = []
        for dir_name in dirs:
            dir_path = root_path / dir_name

            # Skip hidden directories (except .git which we already handled)
            if exclude_hidden_dirs and dir_name.startswith("."):
                dirs_to_remove.append(dir_name)
                continue

            # Check exclusion patterns
            for pattern in exclude_patterns:
                if dir_path.match(pattern):
                    dirs_to_remove.append(dir_name)
                    break

        # Remove excluded directories from walk
        for dir_name in dirs_to_remove:
            dirs.remove(dir_name)

    logger.info(f"Discovered {len(git_repos)} git repositories in {documents_path}")
    return git_repos


def get_commits_after_timestamp(
    git_dir: Path,
    after_timestamp: int | None = None,
) -> list[str]:
    """
    Get commit hashes after a timestamp.

    Args:
        git_dir: Path to .git directory
        after_timestamp: Unix timestamp (None = all commits)

    Returns:
        List of commit SHAs (newest first)
    """
    repo_path = git_dir.parent

    # Build git log command
    cmd = ["git", "log", "--all", "--format=%H"]

    if after_timestamp is not None:
        cmd.append(f"--after={after_timestamp}")

    try:
        result = subprocess.run(
            cmd,
            cwd=repo_path,
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        )

        commit_hashes = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        logger.debug(f"Found {len(commit_hashes)} commits in {repo_path.name}")
        return commit_hashes

    except subprocess.CalledProcessError as e:
        logger.error(f"Git log failed for {repo_path}: {e.stderr}")
        return []
    except subprocess.TimeoutExpired:
        logger.error(f"Git log timeout for {repo_path}")
        return []


def is_git_available() -> bool:
    """Check if git binary is available in PATH."""
    try:
        subprocess.run(
            ["git", "--version"],
            capture_output=True,
            check=True,
            timeout=5,
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return False
