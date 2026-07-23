"""Unit tests for CacheStore adapters (memory_lru and sqlite)."""

from pathlib import Path

from searchkernel.adapters.cache.memory_lru import MemoryLRUCacheStore
from searchkernel.adapters.cache.sqlite import SQLiteCacheStore


class TestMemoryLRUCacheStore:
    """Tests for in-memory LRU cache store."""

    def test_get_set_roundtrip(self):
        """Test basic get/set functionality."""
        cache = MemoryLRUCacheStore(max_entries=10)

        value = {"result": [1, 2, 3]}
        cache.set("test_key", value, epoch=1)

        retrieved = cache.get("test_key")
        assert retrieved == value

    def test_get_nonexistent_returns_none(self):
        """Test that getting a nonexistent key returns None."""
        cache = MemoryLRUCacheStore(max_entries=10)

        assert cache.get("missing_key") is None

    def test_lru_eviction(self):
        """Test that LRU entries are evicted when capacity is exceeded."""
        cache = MemoryLRUCacheStore(max_entries=3)

        # Insert 3 entries
        cache.set("key1", "value1", epoch=1)
        cache.set("key2", "value2", epoch=1)
        cache.set("key3", "value3", epoch=1)

        # All should be present
        assert cache.get("key1") == "value1"
        assert cache.get("key2") == "value2"
        assert cache.get("key3") == "value3"

        # Insert a 4th entry (should evict key1, the least recently used)
        cache.set("key4", "value4", epoch=1)

        # key1 should be evicted
        assert cache.get("key1") is None
        assert cache.get("key2") == "value2"
        assert cache.get("key3") == "value3"
        assert cache.get("key4") == "value4"

    def test_get_updates_lru_order(self):
        """Test that get() moves an entry to the end (most recently used)."""
        cache = MemoryLRUCacheStore(max_entries=3)

        cache.set("key1", "value1", epoch=1)
        cache.set("key2", "value2", epoch=1)
        cache.set("key3", "value3", epoch=1)

        # Access key1, making it recently used
        _ = cache.get("key1")

        # Insert key4 (should evict key2, not key1)
        cache.set("key4", "value4", epoch=1)

        assert cache.get("key1") == "value1"
        assert cache.get("key2") is None
        assert cache.get("key3") == "value3"
        assert cache.get("key4") == "value4"

    def test_invalidate_epoch(self):
        """Test invalidating entries by epoch threshold."""
        cache = MemoryLRUCacheStore(max_entries=10)

        # Insert entries with different epochs
        cache.set("key1", "value1", epoch=1)
        cache.set("key2", "value2", epoch=2)
        cache.set("key3", "value3", epoch=3)

        # Invalidate all entries with epoch <= 2
        cache.invalidate_epoch(2)

        # key1 and key2 should be gone
        assert cache.get("key1") is None
        assert cache.get("key2") is None
        # key3 should still exist
        assert cache.get("key3") == "value3"

    def test_invalidate_epoch_boundary(self):
        """Test that invalidate_epoch uses <= comparison."""
        cache = MemoryLRUCacheStore(max_entries=10)

        cache.set("key1", "value1", epoch=5)
        cache.set("key2", "value2", epoch=6)

        # Invalidate epoch 5 (should remove key1 but not key2)
        cache.invalidate_epoch(5)

        assert cache.get("key1") is None
        assert cache.get("key2") == "value2"

    def test_set_overwrites_existing_key(self):
        """Test that set() overwrites existing keys."""
        cache = MemoryLRUCacheStore(max_entries=10)

        cache.set("key1", "old_value", epoch=1)
        assert cache.get("key1") == "old_value"

        cache.set("key1", "new_value", epoch=2)
        assert cache.get("key1") == "new_value"

    def test_complex_values(self):
        """Test caching complex nested structures."""
        cache = MemoryLRUCacheStore(max_entries=10)

        complex_value = {
            "list": [1, 2, 3],
            "nested": {"a": 1, "b": [4, 5, 6]},
            "string": "test",
        }

        cache.set("complex", complex_value, epoch=1)
        retrieved = cache.get("complex")

        assert retrieved == complex_value
        assert retrieved["nested"]["b"] == [4, 5, 6]


class TestSQLiteCacheStore:
    """Tests for SQLite-backed cache store."""

    def test_get_set_roundtrip(self, tmp_path: Path):
        """Test basic get/set functionality."""
        db_path = tmp_path / "cache.db"
        cache = SQLiteCacheStore(db_path)

        value = {"result": [1, 2, 3]}
        cache.set("test_key", value, epoch=1)

        retrieved = cache.get("test_key")
        assert retrieved == value

    def test_get_nonexistent_returns_none(self, tmp_path: Path):
        """Test that getting a nonexistent key returns None."""
        db_path = tmp_path / "cache.db"
        cache = SQLiteCacheStore(db_path)

        assert cache.get("missing_key") is None

    def test_persistence_across_instances(self, tmp_path: Path):
        """Test that data persists across different store instances."""
        db_path = tmp_path / "cache.db"

        # Create first instance and set a value
        cache1 = SQLiteCacheStore(db_path)
        cache1.set("persist_key", "persist_value", epoch=1)

        # Create second instance and verify the value persists
        cache2 = SQLiteCacheStore(db_path)
        assert cache2.get("persist_key") == "persist_value"

    def test_invalidate_epoch(self, tmp_path: Path):
        """Test invalidating entries by epoch threshold."""
        db_path = tmp_path / "cache.db"
        cache = SQLiteCacheStore(db_path)

        # Insert entries with different epochs
        cache.set("key1", "value1", epoch=1)
        cache.set("key2", "value2", epoch=2)
        cache.set("key3", "value3", epoch=3)

        # Invalidate all entries with epoch <= 2
        cache.invalidate_epoch(2)

        # key1 and key2 should be gone
        assert cache.get("key1") is None
        assert cache.get("key2") is None
        # key3 should still exist
        assert cache.get("key3") == "value3"

    def test_invalidate_epoch_boundary(self, tmp_path: Path):
        """Test that invalidate_epoch uses <= comparison."""
        db_path = tmp_path / "cache.db"
        cache = SQLiteCacheStore(db_path)

        cache.set("key1", "value1", epoch=5)
        cache.set("key2", "value2", epoch=6)

        # Invalidate epoch 5 (should remove key1 but not key2)
        cache.invalidate_epoch(5)

        assert cache.get("key1") is None
        assert cache.get("key2") == "value2"

    def test_set_overwrites_existing_key(self, tmp_path: Path):
        """Test that set() overwrites existing keys."""
        db_path = tmp_path / "cache.db"
        cache = SQLiteCacheStore(db_path)

        cache.set("key1", "old_value", epoch=1)
        assert cache.get("key1") == "old_value"

        cache.set("key1", "new_value", epoch=2)
        assert cache.get("key1") == "new_value"

    def test_complex_values(self, tmp_path: Path):
        """Test caching complex nested structures."""
        db_path = tmp_path / "cache.db"
        cache = SQLiteCacheStore(db_path)

        complex_value = {
            "list": [1, 2, 3],
            "nested": {"a": 1, "b": [4, 5, 6]},
            "string": "test",
        }

        cache.set("complex", complex_value, epoch=1)
        retrieved = cache.get("complex")

        assert retrieved == complex_value
        assert retrieved["nested"]["b"] == [4, 5, 6]

    def test_db_path_creation(self, tmp_path: Path):
        """Test that parent directories are created if they don't exist."""
        db_path = tmp_path / "subdir" / "nested" / "cache.db"

        # Should not raise even though parents don't exist
        cache = SQLiteCacheStore(db_path)

        # Should be able to use it
        cache.set("test", "value", epoch=1)
        assert cache.get("test") == "value"
        assert db_path.exists()
