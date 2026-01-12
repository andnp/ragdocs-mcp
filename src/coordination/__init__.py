from enum import Enum

from src.coordination.singleton import SingletonGuard
from src.coordination.file_lock import IndexLock


class CoordinationMode(Enum):
    SINGLETON = "singleton"
    FILE_LOCK = "file_lock"


__all__ = ["CoordinationMode", "SingletonGuard", "IndexLock"]
