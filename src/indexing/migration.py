"""Backwards-compatibility migration for legacy index formats.

Detects and cleans up index artefacts left by previous architectures:

* **Whoosh keyword directory** (``<index_path>/keyword/``) – written by the
  pre-SQLite keyword index.  The new SQLite-backed ``KeywordIndex`` stores its
  data in ``<index_path>/index.db`` and ignores the old directory, but the
  directory continues to occupy disk space.

* **Snapshot directories** (``<index_path>/snapshots/``) – written by the
  former multiprocess worker that published versioned index snapshots to
  coordinate with the read-only MCP server process.  They are never read by
  the current single-process architecture.

When either is detected the function removes the stale paths and, if Whoosh
data was removed, resets the manifest's ``indexed_files`` map so that the
reconciler re-indexes all documents.  This repopulates the keyword FTS5 table
while reusing the existing FAISS vector index (the on-disk format is
unchanged).
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)

# Files that Whoosh writes inside its index directory.
_WHOOSH_MARKERS = frozenset(["_MAIN_WRITELOCK", "MAIN_WRITELOCK"])


def _is_whoosh_directory(path: Path) -> bool:
    """Return True if *path* looks like an old Whoosh index directory.

    We require:
    - the directory exists
    - it does NOT already contain a modern ``index.db`` (already migrated)
    - it contains at least one file whose name starts with ``MAIN_`` or
      ``_MAIN_`` (Whoosh segment / TOC naming convention)
    """
    if not path.is_dir():
        return False
    if (path / "index.db").exists():
        return False
    return any(
        f.name.startswith("MAIN_") or f.name.startswith("_MAIN_")
        for f in path.iterdir()
        if f.is_file()
    )


def _clear_manifest_indexed_files(index_path: Path) -> None:
    """Reset the ``indexed_files`` map in the saved manifest.

    Clearing the map causes the reconciler to treat every document as new and
    re-index it.  The other manifest fields (embedding model, parser versions,
    chunking config) are preserved so we avoid a more expensive full rebuild.
    """
    from src.indexing.manifest import load_manifest, save_manifest

    manifest = load_manifest(index_path)
    if manifest is None:
        return
    manifest.indexed_files = {}
    save_manifest(index_path, manifest)
    logger.info("Cleared manifest indexed_files to trigger keyword re-indexing")


def detect_and_migrate_legacy_index(index_path: Path) -> bool:
    """Detect and remove stale artefacts from legacy index formats.

    Called once during application startup, before indices are loaded.

    Returns:
        ``True`` if any migration was performed, ``False`` otherwise.
    """
    migrated = False

    # ── 1. Old multiprocess snapshot directories ──────────────────────────
    snapshots_dir = index_path / "snapshots"
    if snapshots_dir.is_dir():
        logger.info(
            "Legacy index snapshots detected at %s – removing (no longer used "
            "in single-process SQLite architecture)",
            snapshots_dir,
        )
        try:
            shutil.rmtree(snapshots_dir)
            logger.info("Removed legacy snapshots directory: %s", snapshots_dir)
            migrated = True
        except OSError as exc:
            logger.warning(
                "Could not remove snapshots directory %s: %s", snapshots_dir, exc
            )

    # ── 2. Old Whoosh keyword index directory ─────────────────────────────
    keyword_dir = index_path / "keyword"
    if _is_whoosh_directory(keyword_dir):
        logger.info(
            "Legacy Whoosh keyword index detected at %s – removing and "
            "resetting manifest so documents are re-indexed into SQLite FTS5",
            keyword_dir,
        )
        try:
            shutil.rmtree(keyword_dir)
            logger.info("Removed legacy Whoosh keyword directory: %s", keyword_dir)
            _clear_manifest_indexed_files(index_path)
            migrated = True
        except OSError as exc:
            logger.warning(
                "Could not remove Whoosh keyword directory %s: %s", keyword_dir, exc
            )

    return migrated
