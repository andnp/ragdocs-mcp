"""File discovery for documents matching parser patterns."""

from __future__ import annotations

import fnmatch
import os
from pathlib import Path

from src.utils import should_include_file

PARSER_SUFFIXES: frozenset[str] = frozenset({".md", ".markdown", ".txt"})


def get_parser_suffixes() -> frozenset[str]:
    return PARSER_SUFFIXES


def is_excluded_dir(
    dir_path: str,
    exclude_patterns: list[str],
    exclude_hidden_dirs: bool,
) -> bool:
    """Check if a directory should be excluded from watching or discovery."""
    normalized = dir_path.replace("\\", "/")
    name = Path(normalized).name

    if exclude_hidden_dirs and name.startswith("."):
        return True

    # Test with a synthetic file path to match directory-level exclude patterns
    test_path = normalized.rstrip("/") + "/test_file"
    for pattern in exclude_patterns:
        if fnmatch.fnmatch(test_path, pattern):
            return True

    return False


def walk_included_dirs(
    root: Path,
    exclude_patterns: list[str],
    exclude_hidden_dirs: bool,
) -> list[Path]:
    """Walk directory tree, returning only non-excluded directories.

    Prunes excluded and hidden directories to avoid unnecessary traversal.
    """
    included: list[Path] = [root]
    for dirpath, dirnames, _ in os.walk(root, topdown=True):
        # Prune in-place so os.walk skips excluded subtrees
        dirnames[:] = [
            d
            for d in dirnames
            if not is_excluded_dir(
                os.path.join(dirpath, d), exclude_patterns, exclude_hidden_dirs
            )
        ]
        for d in dirnames:
            included.append(Path(dirpath) / d)
    return included


def walk_dirs_with_files(
    root: Path,
    exclude_patterns: list[str],
    exclude_hidden_dirs: bool,
    suffixes: set[str] | frozenset[str],
) -> list[Path]:
    """Walk directory tree, returning directories that contain at least one
    parseable file (matching *suffixes*), plus the root itself.

    The root is always included so that files added directly to the documents
    root — even when it is initially empty — are detected immediately.

    Subdirectories are included only if they currently contain a parseable file,
    avoiding inode exhaustion in monorepos where most directories hold source
    code rather than documentation.  New subdirectories are picked up after the
    next reconciliation cycle via ``FileWatcher.refresh_watches()``.
    """
    result: list[Path] = [root]
    for dirpath, dirnames, filenames in os.walk(root, topdown=True):
        # Prune in-place so os.walk skips excluded subtrees
        dirnames[:] = [
            d
            for d in dirnames
            if not is_excluded_dir(
                os.path.join(dirpath, d), exclude_patterns, exclude_hidden_dirs
            )
        ]
        current = Path(dirpath)
        if current != root and any(
            Path(f).suffix.lower() in suffixes for f in filenames
        ):
            result.append(current)
    return result


def discover_files(
    documents_path: str | Path,
    *,
    include_patterns: list[str] | None = None,
    exclude_patterns: list[str] | None = None,
    exclude_hidden_dirs: bool = True,
) -> list[str]:
    """Discover all indexable files matching parser patterns.

    Uses os.walk with directory pruning instead of glob.glob to avoid
    traversing excluded directories (e.g. .venv/, node_modules/).

    Args:
        documents_path: Root directory to search
        include_patterns: Glob patterns for files to include (default: ["*"] = all)
        exclude_patterns: Glob patterns for files to exclude
        exclude_hidden_dirs: Whether to exclude hidden directories (default: True)

    Returns:
        List of absolute file paths to index
    """
    docs_path = Path(documents_path)
    include = include_patterns if include_patterns else ["*"]
    exclude = exclude_patterns or []
    suffixes = get_parser_suffixes()

    all_files: set[str] = set()

    included_dirs = walk_included_dirs(docs_path, exclude, exclude_hidden_dirs)
    for dir_path in included_dirs:
        try:
            for entry in os.scandir(str(dir_path)):
                if entry.is_file() and Path(entry.name).suffix.lower() in suffixes:
                    all_files.add(entry.path)
        except OSError:
            pass

    return [
        f
        for f in sorted(all_files)
        if should_include_file(f, include, exclude, exclude_hidden_dirs)
    ]


def discover_files_multi_root(
    documents_paths: list[str | Path],
    *,
    include_patterns: list[str] | None = None,
    exclude_patterns: list[str] | None = None,
    exclude_hidden_dirs: bool = True,
) -> list[str]:
    all_files: set[str] = set()

    for root in documents_paths:
        path = Path(root)
        if not path.exists():
            continue
        discovered = discover_files(
            path,
            include_patterns=include_patterns,
            exclude_patterns=exclude_patterns,
            exclude_hidden_dirs=exclude_hidden_dirs,
        )
        all_files.update(discovered)

    return sorted(all_files)
