"""Utilities for reading text files with encoding fallback."""

from pathlib import Path


def read_text_with_encoding_fallback(file_path: str | Path) -> tuple[str, str]:
    """Read text file with UTF-8 first, fall back to latin-1/cp1252/iso-8859-1.

    Args:
        file_path: Path to the file to read.

    Returns:
        Tuple of (content, encoding_used).

    Raises:
        UnicodeDecodeError: If no encoding in the fallback list succeeds.
    """
    path = Path(file_path)

    for encoding in ["utf-8", "latin-1", "cp1252", "iso-8859-1"]:
        try:
            content = path.read_text(encoding=encoding, errors="strict")
            return content, encoding
        except (UnicodeDecodeError, LookupError):
            continue

    # All encodings failed
    raise UnicodeDecodeError(
        "utf-8",
        b"",
        0,
        1,
        f"Could not decode {file_path} with any supported encoding",
    )
