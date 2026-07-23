"""Epoch-aware caching decorator and utilities."""

import hashlib
import json
import logging
from functools import wraps
from typing import Any, Callable, TypeVar

from searchkernel.ports.stores import CacheStore

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


def _serialize_key_args(*args: Any, **kwargs: Any) -> str:
    """Serialize function arguments into a stable, hashable string.

    Args:
        *args: Positional arguments to serialize.
        **kwargs: Keyword arguments to serialize.

    Returns:
        A stable JSON string representation of the arguments.
    """
    try:
        # Convert args and kwargs to serializable form
        serializable = {
            "args": args,
            "kwargs": kwargs,
        }
        json_str = json.dumps(serializable, sort_keys=True, default=str)
        return json_str
    except (TypeError, ValueError):
        # Fallback for non-serializable types
        return repr((args, kwargs))


def _make_cache_key(prefix: str, *args: Any, epoch: int, **kwargs: Any) -> str:
    """Create a cache key from a prefix, arguments, and epoch.

    Args:
        prefix: Key prefix (e.g., function name).
        *args: Function arguments.
        epoch: Index epoch.
        **kwargs: Function keyword arguments.

    Returns:
        A cache key including epoch for proper invalidation.
    """
    args_str = _serialize_key_args(*args, **kwargs)
    args_hash = hashlib.sha256(args_str.encode()).hexdigest()[:16]
    return f"{prefix}:{epoch}:{args_hash}"


def get_or_compute(
    store: CacheStore,
    key_prefix: str,
    epoch: int,
    compute_fn: Callable[..., Any],
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Get a value from cache or compute it if not found or stale.

    Args:
        store: CacheStore to use for lookups/stores.
        key_prefix: Prefix for the cache key (e.g., function name).
        epoch: Current index epoch. Keys from older epochs are stale.
        compute_fn: Function to call if cache miss.
        *args: Arguments to pass to compute_fn.
        **kwargs: Keyword arguments to pass to compute_fn.

    Returns:
        Cached value if found and valid, otherwise the result of compute_fn.
    """
    cache_key = _make_cache_key(key_prefix, *args, epoch=epoch, **kwargs)

    # Try to get from cache
    cached = store.get(cache_key)
    if cached is not None:
        logger.debug(f"Cache hit: {key_prefix}")
        return cached

    # Compute and cache
    logger.debug(f"Cache miss: {key_prefix}, computing...")
    result = compute_fn(*args, **kwargs)
    store.set(cache_key, result, epoch)
    return result


def cached(key_prefix: str) -> Callable[[F], F]:
    """Decorator for caching function results with epoch-based invalidation.

    Wraps a function to cache its results. The cache key is built from
    the function arguments (serialized and hashed) and the current epoch.
    When the epoch changes, old cached values are effectively stale
    (due to key mismatch with newer epoch).

    To use, inject the CacheStore at call time or bind it to the decorated
    function.

    Args:
        key_prefix: Prefix for cache keys (typically the function name).

    Returns:
        A decorator that wraps the function.

    Example:
        @cached("my_search_query")
        def search(query: str, k: int) -> list[Result]:
            ...

        # Caller must provide a CacheStore and epoch
        result = search(query="test", k=10)
        # To use with caching, call via get_or_compute instead:
        result = get_or_compute(
            store, "my_search_query", epoch, search, query="test", k=10
        )
    """

    def decorator(fn: F) -> F:
        @wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # This wrapper is a no-op; the actual caching happens via get_or_compute
            # The decorator serves as a marker and documentation.
            return fn(*args, **kwargs)

        wrapper._cache_key_prefix = key_prefix  # type: ignore
        return wrapper  # type: ignore

    return decorator
