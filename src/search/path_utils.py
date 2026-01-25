import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def normalize_path(file_path: str, docs_root: Path):
    path = Path(file_path)

    if path.is_absolute():
        try:
            path = path.relative_to(docs_root)
        except ValueError:
            pass

    return str(path.with_suffix(""))


def matches_any_excluded(file_path: str, excluded_files: set[str], docs_root: Path):
    normalized = normalize_path(file_path, docs_root)

    if normalized in excluded_files:
        return True

    filename = Path(normalized).name
    if filename in excluded_files:
        return True

    return False


def compute_doc_id(file_path: Path, docs_root: Path) -> str:
    """
    Compute document ID from file path relative to docs root.

    Args:
        file_path: Absolute path to the document file
        docs_root: Root directory for documentation

    Returns:
        Document ID (relative path with forward slashes, no extension)

    Example:
        >>> compute_doc_id(Path("/docs/guide/setup.md"), Path("/docs"))
        "guide/setup"
    """
    try:
        rel_path = file_path.relative_to(docs_root)
        # Remove extension and convert to forward slashes
        doc_id = str(rel_path.with_suffix("")).replace("\\", "/")
        return doc_id
    except ValueError:
        # file_path not under docs_root, use absolute path
        logger.warning(
            f"File {file_path} is outside docs root {docs_root}. "
            "Using absolute path as doc_id."
        )
        return str(file_path.with_suffix("")).replace("\\", "/")


def extract_doc_id_from_chunk_id(chunk_id: str) -> str:
    """
    Extract document ID from chunk ID.

    Handles both formats:
    - "doc/path#chunk_0" → "doc/path"
    - "doc_path_chunk_0" → "doc_path"

    Args:
        chunk_id: Chunk identifier (with separator)

    Returns:
        Document ID (without chunk suffix)

    Example:
        >>> extract_doc_id_from_chunk_id("guide/setup#chunk_0")
        "guide/setup"
        >>> extract_doc_id_from_chunk_id("guide_setup_chunk_0")
        "guide_setup"
    """
    # Try hash separator first (preferred format)
    if "#" in chunk_id:
        return chunk_id.split("#")[0]

    # Fall back to underscore separator (legacy format)
    # Split from right, remove last part if it matches "chunk_N"
    parts = chunk_id.rsplit("_", 2)  # ["doc", "chunk", "0"]
    if len(parts) >= 3 and parts[-2] == "chunk":
        return "_".join(parts[:-2])

    # No valid separator found, return as-is
    logger.warning(f"Chunk ID '{chunk_id}' has unexpected format")
    return chunk_id


def resolve_doc_path(
    doc_id: str,
    docs_root: Path,
    extensions: list[str] | None = None,
) -> Path | None:
    """
    Resolve document ID back to absolute file path.

    Args:
        doc_id: Document identifier (relative path without extension)
        docs_root: Root directory for documentation
        extensions: File extensions to try (default: [".md", ".txt"])

    Returns:
        Absolute path if file exists, None otherwise

    Example:
        >>> resolve_doc_path("guide/setup", Path("/docs"))
        Path("/docs/guide/setup.md")  # if exists
    """
    if extensions is None:
        extensions = [".md", ".txt"]

    # Normalize doc_id (handle both forward and back slashes)
    normalized_id = doc_id.replace("\\", "/")

    for ext in extensions:
        candidate = docs_root / normalized_id
        candidate = candidate.with_suffix(ext)

        if candidate.exists() and candidate.is_file():
            return candidate.resolve()

    return None
