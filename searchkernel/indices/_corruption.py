"""Shared SQLite corruption detection for index self-healing."""

_CORRUPTION_PATTERNS = (
    "database disk image is malformed",
    "disk i/o error",
    "file is not a database",
    "unable to open database file",
)

_TRANSIENT_PATTERNS = (
    "database is locked",
    "database table is locked",
    "database schema is locked",
)


def is_corruption_error(exc: Exception) -> bool:
    """Check if an exception indicates SQLite database corruption."""
    msg = str(exc).lower()
    return any(pat in msg for pat in _CORRUPTION_PATTERNS)


def is_transient_error(exc: Exception) -> bool:
    """Check if an exception indicates transient SQLite lock contention."""
    msg = str(exc).lower()
    return any(pat in msg for pat in _TRANSIENT_PATTERNS)
