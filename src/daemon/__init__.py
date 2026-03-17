"""Global daemon scaffolding for Ragdocs V2."""

from src.daemon.lock import DaemonLockTimeoutError, FilesystemLock
from src.daemon.metadata import (
    DaemonMetadata,
    read_daemon_metadata,
    remove_daemon_metadata,
    write_daemon_metadata,
)
from src.daemon.paths import RuntimePaths

__all__ = [
    "DaemonLockTimeoutError",
    "DaemonMetadata",
    "FilesystemLock",
    "RuntimePaths",
    "read_daemon_metadata",
    "remove_daemon_metadata",
    "write_daemon_metadata",
]