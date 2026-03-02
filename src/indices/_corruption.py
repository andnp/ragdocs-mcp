"""Shared SQLite corruption detection for index self-healing."""

_CORRUPTION_PATTERNS = (
    "database disk image is malformed",
    "disk i/o error",
    "file is not a database",
    "database is locked",
    "unable to open database file",
)


def is_corruption_error(exc: Exception) -> bool:
    """Check if an exception indicates SQLite database corruption."""
    msg = str(exc).lower()
    return any(pat in msg for pat in _CORRUPTION_PATTERNS)
