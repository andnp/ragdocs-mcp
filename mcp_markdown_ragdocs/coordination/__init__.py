from enum import Enum

from mcp_markdown_ragdocs.coordination.file_lock import IndexLock
from mcp_markdown_ragdocs.coordination.singleton import SingletonGuard


class CoordinationMode(Enum):
    SINGLETON = "singleton"
    FILE_LOCK = "file_lock"


__all__ = ["CoordinationMode", "IndexLock", "SingletonGuard"]
