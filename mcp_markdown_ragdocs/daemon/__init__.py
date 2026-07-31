"""Global daemon scaffolding for Ragdocs V2."""

from mcp_markdown_ragdocs.daemon.lock import DaemonLockTimeoutError, FilesystemLock
from mcp_markdown_ragdocs.daemon.metadata import (
    DaemonMetadata,
    read_daemon_metadata,
    remove_daemon_metadata,
    write_daemon_metadata,
)
from mcp_markdown_ragdocs.daemon.paths import RuntimePaths

__all__ = [
    "DaemonLockTimeoutError",
    "DaemonMetadata",
    "FilesystemLock",
    "RuntimePaths",
    "read_daemon_metadata",
    "remove_daemon_metadata",
    "write_daemon_metadata",
]