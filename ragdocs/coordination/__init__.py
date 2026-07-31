from enum import Enum

from searchkernel.coordination.file_lock import IndexLock
from searchkernel.coordination.singleton import SingletonGuard


class CoordinationMode(Enum):
    SINGLETON = "singleton"
    FILE_LOCK = "file_lock"


__all__ = ["CoordinationMode", "IndexLock", "SingletonGuard"]
