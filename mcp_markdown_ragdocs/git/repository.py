"""Git repository discovery and commit listing."""

import logging
import os
import subprocess
from collections.abc import Generator
from pathlib import Path

from searchkernel.api import is_excluded_dir

logger = logging.getLogger(__name__)


def discover_git_repositories(
    documents_path: Path,
    exclude_patterns: list[str],
    exclude_hidden_dirs: bool = True,
) -> list[Path]:
    """
    Recursively discover .git directories.

    Uses is_excluded_dir() for consistent pattern matching with file discovery.

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
            # .git dirs are the target — only exclude if an explicit pattern
            # matches the repo root itself (not the .git contents pattern)
            if not is_excluded_dir(
                str(root_path), exclude_patterns, exclude_hidden_dirs
            ):
                git_repos.append(git_dir.resolve())
                logger.debug(f"Found git repository: {git_dir}")

            # Don't descend into this directory further
            dirs.clear()
            continue

        # Prune excluded directories in-place
        dirs[:] = [
            d
            for d in dirs
            if not is_excluded_dir(
                os.path.join(root, d), exclude_patterns, exclude_hidden_dirs
            )
        ]

    logger.info(f"Discovered {len(git_repos)} git repositories in {documents_path}")
    return git_repos


def discover_git_repositories_multi_root(
    documents_paths: list[Path],
    exclude_patterns: list[str],
    exclude_hidden_dirs: bool = True,
) -> list[Path]:
    discovered: set[Path] = set()

    for root in documents_paths:
        if not root.exists():
            continue
        discovered.update(
            discover_git_repositories(root, exclude_patterns, exclude_hidden_dirs)
        )

    return sorted(discovered)


def iter_commit_hashes_after_timestamp(
    git_dir: Path,
    after_timestamp: int | None = None,
) -> Generator[str]:
    """
    Yield commit hashes after a timestamp.

    Args:
        git_dir: Path to .git directory
        after_timestamp: Unix timestamp (None = all commits)

    Raises:
        subprocess.CalledProcessError: If git log exits unsuccessfully
        subprocess.TimeoutExpired: If git log does not finish within 30 seconds
    """
    repo_path = git_dir.parent

    cmd = ["git", "log", "--all", "--format=%H"]

    if after_timestamp is not None:
        cmd.append(f"--after={after_timestamp}")

    process = subprocess.Popen(
        cmd,
        cwd=repo_path,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    try:
        if process.stdout is None:
            raise RuntimeError("Git log did not provide stdout")

        for line in process.stdout:
            commit_hash = line.strip()
            if commit_hash:
                yield commit_hash

        returncode = process.wait(timeout=30)
        if returncode != 0:
            stderr = process.stderr.read() if process.stderr is not None else ""
            raise subprocess.CalledProcessError(
                returncode,
                cmd,
                stderr=stderr,
            )
    finally:
        if process.poll() is None:
            process.kill()
            try:
                process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                logger.warning("Git log process did not terminate for %s", repo_path)

        if process.stdout is not None:
            process.stdout.close()
        if process.stderr is not None:
            process.stderr.close()


def get_commits_after_timestamp(
    git_dir: Path,
    after_timestamp: int | None = None,
) -> list[str]:
    """Get commit hashes after a timestamp as a list."""
    repo_path = git_dir.parent

    try:
        commit_hashes = list(
            iter_commit_hashes_after_timestamp(git_dir, after_timestamp)
        )
        logger.debug(f"Found {len(commit_hashes)} commits in {repo_path.name}")
        return commit_hashes
    except subprocess.CalledProcessError as e:
        logger.error(f"Git log failed for {repo_path}: {e.stderr}")
        return []
    except subprocess.TimeoutExpired:
        logger.error(f"Git log timeout for {repo_path}")
        return []


_REF_SIGNATURE_CACHE: dict[Path, tuple[int, str | None]] = {}


def get_git_refs_mtime_ns(git_dir: Path) -> int:
    """Return the max mtime of git ref containers (.git/HEAD, .git/packed-refs, .git/refs)."""
    mtime = 0
    for entry in ("HEAD", "packed-refs"):
        try:
            mtime = max(mtime, (git_dir / entry).stat().st_mtime_ns)
        except OSError:
            pass
    refs_dir = git_dir / "refs"
    if refs_dir.is_dir():
        try:
            mtime = max(mtime, refs_dir.stat().st_mtime_ns)
            for root, _, files in os.walk(refs_dir):
                for f in files:
                    try:
                        mtime = max(mtime, os.stat(os.path.join(root, f)).st_mtime_ns)
                    except OSError:
                        pass
        except OSError:
            pass
    return mtime


def get_git_ref_signature(git_dir: Path) -> str | None:
    """Return a stable signature for the refs visible to ``git log --all``."""
    resolved_git_dir = git_dir.resolve()
    current_mtime = get_git_refs_mtime_ns(resolved_git_dir)
    cached = _REF_SIGNATURE_CACHE.get(resolved_git_dir)
    if cached is not None and cached[0] == current_mtime:
        return cached[1]

    repo_path = resolved_git_dir.parent
    try:
        head = subprocess.run(
            ["git", "rev-parse", "--verify", "HEAD"],
            cwd=repo_path,
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        ).stdout.strip()
        refs = subprocess.run(
            [
                "git",
                "for-each-ref",
                "--format=%(refname)=%(objectname)",
                "refs/heads",
                "refs/remotes",
                "refs/tags",
            ],
            cwd=repo_path,
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        ).stdout
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        _REF_SIGNATURE_CACHE[resolved_git_dir] = (current_mtime, None)
        return None

    signature = f"{head}\n{refs}"
    _REF_SIGNATURE_CACHE[resolved_git_dir] = (current_mtime, signature)
    return signature


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
    except (
        subprocess.CalledProcessError,
        FileNotFoundError,
        subprocess.TimeoutExpired,
    ):
        return False
