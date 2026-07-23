"""Unit tests for the @cached decorator and get_or_compute helper."""

from pathlib import Path
from typing import Any

from searchkernel.adapters.cache.memory_lru import MemoryLRUCacheStore
from searchkernel.adapters.cache.sqlite import SQLiteCacheStore
from searchkernel.runtime.cache import cached, get_or_compute


class TestGetOrCompute:
    """Tests for the get_or_compute helper function."""

    def test_cache_hit(self):
        """Test that cached values are returned on hit."""
        store = MemoryLRUCacheStore(max_entries=10)
        call_count = 0

        def compute_fn(x: int, y: int) -> int:
            nonlocal call_count
            call_count += 1
            return x + y

        epoch = 1
        # First call should compute
        result1 = get_or_compute(
            store, "add", epoch, compute_fn, 2, 3
        )
        assert result1 == 5
        assert call_count == 1

        # Second call should hit cache
        result2 = get_or_compute(
            store, "add", epoch, compute_fn, 2, 3
        )
        assert result2 == 5
        assert call_count == 1  # Not incremented

    def test_cache_miss_different_args(self):
        """Test that different arguments cause cache miss."""
        store = MemoryLRUCacheStore(max_entries=10)
        call_count = 0

        def compute_fn(x: int) -> int:
            nonlocal call_count
            call_count += 1
            return x * 2

        epoch = 1
        result1 = get_or_compute(store, "double", epoch, compute_fn, 5)
        assert result1 == 10
        assert call_count == 1

        # Different argument
        result2 = get_or_compute(store, "double", epoch, compute_fn, 7)
        assert result2 == 14
        assert call_count == 2

    def test_epoch_bump_invalidates(self):
        """Test that bumping the epoch invalidates old cached values."""
        store = MemoryLRUCacheStore(max_entries=10)
        call_count = 0

        def compute_fn(x: int) -> int:
            nonlocal call_count
            call_count += 1
            return x * 2

        # First call with epoch 1
        result1 = get_or_compute(store, "double", 1, compute_fn, 5)
        assert result1 == 10
        assert call_count == 1

        # Same call with epoch 1 (should hit cache)
        result2 = get_or_compute(store, "double", 1, compute_fn, 5)
        assert result2 == 10
        assert call_count == 1

        # Same call with epoch 2 (should miss cache due to different key)
        result3 = get_or_compute(store, "double", 2, compute_fn, 5)
        assert result3 == 10
        assert call_count == 2

    def test_kwargs_affect_cache_key(self):
        """Test that keyword arguments affect the cache key."""
        store = MemoryLRUCacheStore(max_entries=10)
        call_count = 0

        def compute_fn(x: int, multiplier: int = 2) -> int:
            nonlocal call_count
            call_count += 1
            return x * multiplier

        epoch = 1
        result1 = get_or_compute(
            store, "multiply", epoch, compute_fn, 5, multiplier=2
        )
        assert result1 == 10
        assert call_count == 1

        # Same call (should hit cache)
        result2 = get_or_compute(
            store, "multiply", epoch, compute_fn, 5, multiplier=2
        )
        assert result2 == 10
        assert call_count == 1

        # Different kwarg (should miss cache)
        result3 = get_or_compute(
            store, "multiply", epoch, compute_fn, 5, multiplier=3
        )
        assert result3 == 15
        assert call_count == 2

    def test_with_sqlite_store(self, tmp_path: Path):
        """Test get_or_compute works with SQLite store."""
        db_path = tmp_path / "cache.db"
        store = SQLiteCacheStore(db_path)
        call_count = 0

        def compute_fn(x: str) -> str:
            nonlocal call_count
            call_count += 1
            return x.upper()

        epoch = 1
        result1 = get_or_compute(store, "uppercase", epoch, compute_fn, "hello")
        assert result1 == "HELLO"
        assert call_count == 1

        # Create new store instance from same DB
        store2 = SQLiteCacheStore(db_path)
        result2 = get_or_compute(store2, "uppercase", epoch, compute_fn, "hello")
        assert result2 == "HELLO"
        assert call_count == 1  # Should hit persisted cache

    def test_complex_return_values(self):
        """Test caching of complex return values."""
        store = MemoryLRUCacheStore(max_entries=10)
        call_count = 0

        def compute_fn(query: str) -> dict[str, Any]:
            nonlocal call_count
            call_count += 1
            return {"query": query, "results": [1, 2, 3]}

        epoch = 1
        result1 = get_or_compute(
            store, "search", epoch, compute_fn, "test"
        )
        assert result1 == {"query": "test", "results": [1, 2, 3]}
        assert call_count == 1

        result2 = get_or_compute(
            store, "search", epoch, compute_fn, "test"
        )
        assert result2 == {"query": "test", "results": [1, 2, 3]}
        assert call_count == 1


class TestCachedDecorator:
    """Tests for the @cached decorator."""

    def test_decorator_marks_function(self):
        """Test that @cached decorator marks the function."""
        @cached("my_fn")
        def my_function(x: int) -> int:
            return x * 2

        assert hasattr(my_function, "_cache_key_prefix")
        assert my_function._cache_key_prefix == "my_fn"

    def test_decorated_function_still_works(self):
        """Test that decorated function is still callable."""
        @cached("my_fn")
        def my_function(x: int) -> int:
            return x * 2

        # Direct call (without cache) should still work
        assert my_function(5) == 10
        assert my_function(7) == 14

    def test_multiple_decorations(self):
        """Test multiple functions can be decorated with different keys."""
        @cached("add_fn")
        def add_fn(a: int, b: int) -> int:
            return a + b

        @cached("mul_fn")
        def mul_fn(a: int, b: int) -> int:
            return a * b

        assert add_fn._cache_key_prefix == "add_fn"
        assert mul_fn._cache_key_prefix == "mul_fn"
        assert add_fn(2, 3) == 5
        assert mul_fn(2, 3) == 6


class TestEpochInvalidation:
    """Integration tests for epoch-based invalidation."""

    def test_invalidate_epoch_removes_stale_entries(self):
        """Test that invalidate_epoch properly removes entries."""
        store = MemoryLRUCacheStore(max_entries=10)
        call_count = 0

        def compute_fn(x: int) -> int:
            nonlocal call_count
            call_count += 1
            return x * 2

        # Cache with epoch 1
        result = get_or_compute(store, "fn", 1, compute_fn, 5)
        assert result == 10
        assert call_count == 1

        # Invalidate epoch 1 and earlier
        store.invalidate_epoch(1)

        # Access same value with same epoch (should recompute
        # because the key includes epoch and old entries are gone)
        result = get_or_compute(store, "fn", 1, compute_fn, 5)
        assert result == 10
        assert call_count == 2

    def test_epoch_progression_scenario(self):
        """Test realistic epoch progression scenario."""
        store = MemoryLRUCacheStore(max_entries=10)
        compute_count = {}

        def make_compute_fn(fn_name: str):
            compute_count[fn_name] = 0

            def fn(x: int) -> int:
                compute_count[fn_name] += 1
                return x + compute_count[fn_name]

            return fn

        # Epoch 1: Cache some results
        fn1 = make_compute_fn("fn1")
        result = get_or_compute(store, "fn1", 1, fn1, 10)
        assert result == 11  # 10 + 1
        assert compute_count["fn1"] == 1

        # Reuse cache in epoch 1
        result = get_or_compute(store, "fn1", 1, fn1, 10)
        assert result == 11
        assert compute_count["fn1"] == 1

        # Epoch 2: Should be a new key, so it computes
        result = get_or_compute(store, "fn1", 2, fn1, 10)
        assert result == 12  # 10 + 2
        assert compute_count["fn1"] == 2

        # Invalidate old epochs (1 and earlier)
        store.invalidate_epoch(1)

        # Epoch 1 again: Should recompute (no old entry)
        result = get_or_compute(store, "fn1", 1, fn1, 10)
        assert result == 13  # 10 + 3
        assert compute_count["fn1"] == 3
