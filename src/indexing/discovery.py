"""File discovery for documents matching parser patterns."""

from __future__ import annotations

import fnmatch
import os
from pathlib import Path

from src.utils import should_include_file

# Default suffixes when no parsers are configured
DEFAULT_SUFFIXES: set[str] = {".md", ".markdown"}


def get_parser_suffixes(
    parsers: dict[str, str],
    fallback: set[str] | None = None,
) -> set[str]:
    """Extract file suffixes from parser glob patterns.

    Args:
        parsers: Mapping of glob patterns to parser names (e.g. {"**/*.md": "MarkdownParser"})
        fallback: Suffixes to use if no suffixes can be extracted from parsers

    Returns:
        Set of lowercase file suffixes (e.g. {".md", ".txt"})
    """
    suffixes: set[str] = set()
    for pattern in parsers:
        suffix = Path(pattern).suffix
        if suffix:
            suffixes.add(suffix.lower())

    if not suffixes:
        return fallback if fallback is not None else set(DEFAULT_SUFFIXES)

    return suffixes


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
            d for d in dirnames
            if not is_excluded_dir(
                os.path.join(dirpath, d), exclude_patterns, exclude_hidden_dirs
            )
        ]
        for d in dirnames:
            included.append(Path(dirpath) / d)
    return included


def discover_files(
    documents_path: str | Path,
    parsers: dict[str, str],
    *,
    recursive: bool = True,
    include_patterns: list[str] | None = None,
    exclude_patterns: list[str] | None = None,
    exclude_hidden_dirs: bool = True,
) -> list[str]:
    """Discover all indexable files matching parser patterns.

    Uses os.walk with directory pruning instead of glob.glob to avoid
    traversing excluded directories (e.g. .venv/, node_modules/).

    Args:
        documents_path: Root directory to search
        parsers: Mapping of glob patterns to parser names
        recursive: Whether to search recursively (default: True)
        include_patterns: Glob patterns for files to include (default: ["*"] = all)
        exclude_patterns: Glob patterns for files to exclude
        exclude_hidden_dirs: Whether to exclude hidden directories (default: True)

    Returns:
        List of absolute file paths to index
    """
    docs_path = Path(documents_path)
    include = include_patterns if include_patterns else ["*"]
    exclude = exclude_patterns or []
    suffixes = get_parser_suffixes(parsers)

    all_files: set[str] = set()

    if recursive:
        included_dirs = walk_included_dirs(docs_path, exclude, exclude_hidden_dirs)
        for dir_path in included_dirs:
            try:
                for entry in os.scandir(str(dir_path)):
                    if entry.is_file() and Path(entry.name).suffix.lower() in suffixes:
                        all_files.add(entry.path)
            except OSError:
                pass
    else:
        # Non-recursive: only scan the root directory
        try:
            for entry in os.scandir(str(docs_path)):
                if entry.is_file() and Path(entry.name).suffix.lower() in suffixes:
                    all_files.add(entry.path)
        except OSError:
            pass

    return [
        f for f in sorted(all_files)
        if should_include_file(f, include, exclude, exclude_hidden_dirs)
    ]
