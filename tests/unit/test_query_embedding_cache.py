import threading
from dataclasses import dataclass

import pytest

from searchkernel.runtime import QueryEmbeddingCache


@dataclass
class _Clock:
    value: float = 0.0

    def __call__(self) -> float:
        return self.value


def test_cache_keys_by_model_and_query_and_returns_copies() -> None:
    cache = QueryEmbeddingCache()
    calls = 0

    def compute() -> list[float]:
        nonlocal calls
        calls += 1
        return [1.0, 2.0]

    first = cache.get_or_compute(model_name="alpha", query="same", compute=compute)
    first.append(3.0)
    second = cache.get_or_compute(model_name="alpha", query="same", compute=compute)
    third = cache.get_or_compute(model_name="beta", query="same", compute=compute)

    assert second == [1.0, 2.0]
    assert third == [1.0, 2.0]
    assert calls == 2


def test_cache_expires_entries() -> None:
    clock = _Clock()
    cache = QueryEmbeddingCache(ttl_seconds=5.0, clock=clock)
    calls = 0

    def compute() -> list[float]:
        nonlocal calls
        calls += 1
        return [float(calls)]

    assert cache.get_or_compute(model_name="model", query="query", compute=compute) == [1.0]
    clock.value = 5.0
    assert cache.get_or_compute(model_name="model", query="query", compute=compute) == [2.0]


def test_cache_evicts_least_recently_used_entries() -> None:
    cache = QueryEmbeddingCache(max_entries=2)
    calls: dict[str, int] = {}

    def compute_for(query: str) -> list[float]:
        calls[query] = calls.get(query, 0) + 1
        return [float(calls[query])]

    cache.get_or_compute(model_name="model", query="a", compute=lambda: compute_for("a"))
    cache.get_or_compute(model_name="model", query="b", compute=lambda: compute_for("b"))
    cache.get_or_compute(model_name="model", query="a", compute=lambda: compute_for("a"))
    cache.get_or_compute(model_name="model", query="c", compute=lambda: compute_for("c"))
    cache.get_or_compute(model_name="model", query="b", compute=lambda: compute_for("b"))

    assert calls == {"a": 1, "b": 2, "c": 1}


def test_cache_coalesces_concurrent_misses() -> None:
    cache = QueryEmbeddingCache()
    started = threading.Event()
    release = threading.Event()
    calls = 0
    results: list[list[float]] = []

    def compute() -> list[float]:
        nonlocal calls
        calls += 1
        started.set()
        assert release.wait(timeout=5.0)
        return [1.0]

    def load() -> None:
        results.append(cache.get_or_compute(model_name="model", query="query", compute=compute))

    leader = threading.Thread(target=load)
    follower = threading.Thread(target=load)
    leader.start()
    assert started.wait(timeout=5.0)
    follower.start()
    release.set()
    leader.join(timeout=5.0)
    follower.join(timeout=5.0)

    assert sorted(results) == [[1.0], [1.0]]
    assert calls == 1


def test_cache_does_not_cache_failures() -> None:
    cache = QueryEmbeddingCache()
    calls = 0

    def fail() -> list[float]:
        nonlocal calls
        calls += 1
        raise RuntimeError("embedding failed")

    with pytest.raises(RuntimeError, match="embedding failed"):
        cache.get_or_compute(model_name="model", query="query", compute=fail)
    with pytest.raises(RuntimeError, match="embedding failed"):
        cache.get_or_compute(model_name="model", query="query", compute=fail)
    assert calls == 2


@pytest.mark.parametrize(
    ("ttl_seconds", "max_entries", "message"),
    [
        (0.0, 128, "ttl_seconds"),
        (300.0, 0, "max_entries"),
    ],
)
def test_cache_validates_bounds(ttl_seconds: float, max_entries: int, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        QueryEmbeddingCache(ttl_seconds=ttl_seconds, max_entries=max_entries)
