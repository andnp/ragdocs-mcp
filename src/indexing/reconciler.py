import logging
from pathlib import Path

from src.indexing.manifest import IndexManifest
from src.utils import should_include_file

logger = logging.getLogger(__name__)


def find_excluded_indexed_files(
    manifest: IndexManifest,
    docs_path: Path,
    include_patterns: list[str],
    exclude_patterns: list[str],
    exclude_hidden_dirs: bool = True,
) -> list[str]:
    """Find indexed files that should now be excluded based on current patterns.

    Scans the manifest's indexed files and checks each against the current
    include/exclude configuration to detect files that were indexed but
    should now be removed due to blacklist changes.

    Returns:
        List of doc_ids that should be removed due to exclusion patterns.
    """
    excluded_doc_ids: list[str] = []
    indexed_files = manifest.indexed_files or {}

    for doc_id, rel_path in indexed_files.items():
        # Reconstruct absolute path for pattern matching
        abs_path = str(docs_path / rel_path)

        if not should_include_file(
            abs_path,
            include_patterns,
            exclude_patterns,
            exclude_hidden_dirs,
        ):
            excluded_doc_ids.append(doc_id)
            logger.info(f"Indexed file now excluded by pattern: {rel_path}")

    if excluded_doc_ids:
        logger.info(
            f"Found {len(excluded_doc_ids)} indexed files that are now excluded"
        )

    return excluded_doc_ids


def reconcile_indices(
    discovered_files: list[str],
    manifest: IndexManifest,
    docs_path: Path,
    include_patterns: list[str] | None = None,
    exclude_patterns: list[str] | None = None,
    exclude_hidden_dirs: bool = True,
) -> tuple[list[str], list[str], dict[str, str]]:
    """Reconcile indices with filesystem state.

    Args:
        discovered_files: List of absolute paths to files currently on disk
        manifest: Current index manifest
        docs_path: Root documents directory
        include_patterns: Patterns for files to include (default: ["**/*"])
        exclude_patterns: Patterns for files to exclude (default: [])
        exclude_hidden_dirs: Whether to exclude hidden directories

    Returns:
        Tuple of (files_to_add, doc_ids_to_remove, moved_files)
        where moved_files maps old_doc_id -> new_file_path
    """
    # First, explicitly check for indexed files that should now be excluded
    # This provides clear logging for blacklist changes
    excluded_doc_ids: set[str] = set()
    if include_patterns is not None and exclude_patterns is not None:
        excluded_doc_ids = set(
            find_excluded_indexed_files(
                manifest,
                docs_path,
                include_patterns,
                exclude_patterns,
                exclude_hidden_dirs,
            )
        )

    # Convert discovered files to doc_ids (relative path without extension)
    discovered_doc_ids: set[str] = set()
    doc_id_to_abs: dict[str, str] = {}
    for abs_path in discovered_files:
        try:
            rel_path = Path(abs_path).relative_to(docs_path)
            doc_id = str(rel_path.with_suffix(""))
            discovered_doc_ids.add(doc_id)
            doc_id_to_abs[doc_id] = abs_path
        except ValueError:
            logger.warning(f"File outside documents path, skipping: {abs_path}")
            continue

    # Get currently indexed doc_ids from manifest
    indexed_files = manifest.indexed_files or {}
    indexed_doc_ids = set(indexed_files.keys())

    # Find stale entries: doc_ids in manifest but not discovered
    # (and not already flagged as excluded - avoid duplicate logging)
    stale_doc_ids: list[str] = []
    for doc_id in indexed_doc_ids:
        if doc_id not in discovered_doc_ids and doc_id not in excluded_doc_ids:
            stale_doc_ids.append(doc_id)
            logger.info(f"Stale entry detected (file missing): {doc_id}")

    # Combine stale and excluded for removal
    all_doc_ids_to_remove = stale_doc_ids + list(excluded_doc_ids)

    # Find new files: discovered but not in manifest
    new_files: list[str] = []
    for doc_id in discovered_doc_ids:
        if doc_id not in indexed_doc_ids:
            new_files.append(doc_id_to_abs[doc_id])
            logger.info(f"New file detected: {doc_id_to_abs[doc_id]}")

    if stale_doc_ids:
        logger.info(f"Reconciliation: {len(stale_doc_ids)} stale entries to remove")
    if excluded_doc_ids:
        logger.info(
            f"Reconciliation: {len(excluded_doc_ids)} excluded entries to remove"
        )
    if new_files:
        logger.info(f"Reconciliation: {len(new_files)} new files to index")
    if not all_doc_ids_to_remove and not new_files:
        logger.debug("Reconciliation: No changes needed")

    # Return empty dict for moves (will be populated by caller)
    return new_files, all_doc_ids_to_remove, {}


def build_indexed_files_map(
    indexed_files: list[str],
    docs_path: Path
) -> dict[str, str]:
    indexed_map: dict[str, str] = {}
    for abs_path in indexed_files:
        try:
            rel_path = Path(abs_path).relative_to(docs_path)
            doc_id = str(rel_path.with_suffix(""))
            # Map doc_id -> relative path (with extension for reference)
            indexed_map[doc_id] = str(rel_path)
        except ValueError:
            logger.warning(f"File outside documents path, skipping: {abs_path}")
            continue

    return indexed_map
