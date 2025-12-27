import logging
from pathlib import Path

from src.indexing.manifest import IndexManifest

logger = logging.getLogger(__name__)


def reconcile_indices(
    discovered_files: list[str],
    manifest: IndexManifest,
    docs_path: Path
) -> tuple[list[str], list[str]]:
    # Convert discovered files to relative paths with doc_id mapping
    discovered_map: dict[str, str] = {}  # relative_path -> absolute_path
    for abs_path in discovered_files:
        try:
            rel_path = str(Path(abs_path).relative_to(docs_path))
            # Generate doc_id same way as in manager (filename without extension)
            doc_id = Path(abs_path).stem
            discovered_map[rel_path] = abs_path
        except ValueError:
            # File is outside docs_path, skip
            logger.warning(f"File outside documents path, skipping: {abs_path}")
            continue

    # Get currently indexed files (doc_id -> relative_path)
    indexed_files = manifest.indexed_files or {}

    # Find stale entries: files in manifest but not discovered
    stale_doc_ids = []
    for doc_id, rel_path in indexed_files.items():
        if rel_path not in discovered_map:
            stale_doc_ids.append(doc_id)
            logger.info(f"Stale entry detected: {doc_id} (path: {rel_path})")

    # Find new files: discovered but not in manifest
    new_files = []
    for rel_path, abs_path in discovered_map.items():
        if rel_path not in indexed_files.values():
            new_files.append(abs_path)
            logger.info(f"New file detected: {abs_path}")

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
            rel_path = str(Path(abs_path).relative_to(docs_path))
            doc_id = Path(abs_path).stem
            indexed_map[doc_id] = rel_path
        except ValueError:
            logger.warning(f"File outside documents path, skipping: {abs_path}")
            continue

    return indexed_map
