"""Global daemon scaffolding for Ragdocs V2."""

from mcp_markdown_ragdocs.daemon.lock import DaemonLockTimeoutError, FilesystemLock
from mcp_markdown_ragdocs.daemon.metadata import (
    DaemonMetadata,
    read_daemon_metadata,
    remove_daemon_metadata,
    write_daemon_metadata,
)
from mcp_markdown_ragdocs.daemon.paths import RuntimePaths
from mcp_markdown_ragdocs.daemon.producer import ProducerMetadata

__all__ = [
    "DaemonLockTimeoutError",
    "DaemonMetadata",
    "FilesystemLock",
    "RuntimePaths",
    "ProducerMetadata",
    "read_daemon_metadata",
    "remove_daemon_metadata",
    "write_daemon_metadata",
]