import logging
from pathlib import Path

from src.indexing.manifest import IndexManifest

logger = logging.getLogger(__name__)


def reconcile_indices(
    discovered_files: list[str],
    manifest: IndexManifest,
    docs_path: Path
) -> tuple[list[str], list[str]]:
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
    stale_doc_ids = []
    for doc_id in indexed_doc_ids:
        if doc_id not in discovered_doc_ids:
            stale_doc_ids.append(doc_id)
            logger.info(f"Stale entry detected: {doc_id}")

    # Find new files: discovered but not in manifest
    new_files = []
    for doc_id in discovered_doc_ids:
        if doc_id not in indexed_doc_ids:
            new_files.append(doc_id_to_abs[doc_id])
            logger.info(f"New file detected: {doc_id_to_abs[doc_id]}")

    if stale_doc_ids:
        logger.info(f"Reconciliation: {len(stale_doc_ids)} stale entries to remove")
    if new_files:
        logger.info(f"Reconciliation: {len(new_files)} new files to index")
    if not stale_doc_ids and not new_files:
        logger.debug("Reconciliation: No changes needed")

    return new_files, stale_doc_ids


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
