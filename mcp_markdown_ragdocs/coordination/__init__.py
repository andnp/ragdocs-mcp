from enum import Enum

from mcp_markdown_ragdocs.coordination.file_lock import IndexLock


class CoordinationMode(Enum):
    SINGLETON = "singleton"
    FILE_LOCK = "file_lock"


__all__ = ["CoordinationMode", "IndexLock"]
