"""In-memory LRU cache store with epoch-based invalidation."""

from collections import OrderedDict
from typing import Any


class MemoryLRUCacheStore:
    """LRU in-memory cache store implementing the CacheStore port.

    Stores key-value pairs in memory with a bounded size. When capacity is
    exceeded, the least recently used entry is evicted. Entries are tagged
    with an epoch for invalidation support.
    """

    def __init__(self, max_entries: int = 128):
        """Initialize memory LRU cache store.

        Args:
            max_entries: Maximum number of entries to store (default 128).
                        Must be at least 1.
        """
        self._max_entries = max(1, max_entries)
        self._entries: OrderedDict[str, tuple[Any, int]] = OrderedDict()

    def get(self, key: str) -> Any | None:
        """Retrieve a cached value.

        Args:
            key: Cache key.

        Returns:
            Cached value, or None if not found.
        """
        if key not in self._entries:
            return None

        value, _epoch = self._entries[key]
        # Move to end to mark as recently used (LRU)
        self._entries.move_to_end(key)
        return value

    def set(self, key: str, value: Any, epoch: int) -> None:
        """Store a value with an associated epoch.

        Args:
            key: Cache key.
            value: Value to cache.
            epoch: Index epoch at cache time. Used for invalidation.
        """
        if key in self._entries:
            self._entries.pop(key)

        self._entries[key] = (value, epoch)
        self._entries.move_to_end(key)

        # Evict LRU entry if over capacity
        while len(self._entries) > self._max_entries:
            self._entries.popitem(last=False)

    def invalidate_epoch(self, epoch: int) -> None:
        """Invalidate all entries from an epoch or earlier.

        Args:
            epoch: Entries with epoch <= this are discarded.
        """
        keys_to_delete = [
            key for key, (_, entry_epoch) in self._entries.items()
            if entry_epoch <= epoch
        ]
        for key in keys_to_delete:
            del self._entries[key]
