"""File discovery for documents matching parser patterns."""

from __future__ import annotations

import glob
from pathlib import Path

from src.utils import should_include_file


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

    # Collect all files matching parser patterns
    all_files: set[str] = set()
    for pattern in parsers.keys():
        glob_pattern = str(docs_path / pattern)
        files = glob.glob(glob_pattern, recursive=recursive)
        all_files.update(files)

    return [
        f for f in sorted(all_files)
        if should_include_file(
            f,
            include,
            exclude,
            exclude_hidden_dirs,
        )
    ]
