from enum import Enum

from searchkernel.coordination.singleton import SingletonGuard
from searchkernel.coordination.file_lock import IndexLock


class CoordinationMode(Enum):
    SINGLETON = "singleton"
    FILE_LOCK = "file_lock"


__all__ = ["CoordinationMode", "SingletonGuard", "IndexLock"]
