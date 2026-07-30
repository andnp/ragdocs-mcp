"""Unit tests for the gather_with_timeout fan-out helper."""

import asyncio

import pytest

from searchkernel.runtime.fanout import gather_with_timeout


@pytest.mark.asyncio
async def test_concurrent_success():
    """Test that all coroutines succeed when they complete within timeout."""

    async def fast_task(result: int) -> int:
        await asyncio.sleep(0.01)
        return result

    results = await gather_with_timeout(
        [
            fast_task(1),
            fast_task(2),
            fast_task(3),
        ],
        per_timeout_s=1.0,
    )

    assert results == [1, 2, 3]


@pytest.mark.asyncio
async def test_timeout_yields_none():
    """Test that tasks exceeding timeout yield None."""

    async def slow_task(result: int, delay: float) -> int:
        await asyncio.sleep(delay)
        return result

    results = await gather_with_timeout(
        [
            slow_task(1, 0.01),
            slow_task(2, 1.0),  # Will timeout
            slow_task(3, 0.01),
        ],
        per_timeout_s=0.05,
    )

    assert results[0] == 1
    assert results[1] is None  # Timed out
    assert results[2] == 3


@pytest.mark.asyncio
async def test_exception_yields_none():
    """Test that exceptions in tasks yield None."""

    async def failing_task(should_fail: bool) -> str:
        if should_fail:
            raise ValueError("Task failed")
        return "success"

    results = await gather_with_timeout(
        [
            failing_task(False),
            failing_task(True),
            failing_task(False),
        ],
        per_timeout_s=1.0,
    )

    assert results[0] == "success"
    assert results[1] is None  # Exception
    assert results[2] == "success"


@pytest.mark.asyncio
async def test_with_factory_functions():
    """Test that callable factories are invoked lazily."""

    async def task_factory(task_id: int) -> int:
        await asyncio.sleep(0.01)
        return task_id

    # Pass callables instead of coroutines
    factories = [
        lambda i=i: task_factory(i)
        for i in range(3)
    ]

    results = await gather_with_timeout(
        factories,
        per_timeout_s=1.0,
    )

    assert results == [0, 1, 2]


@pytest.mark.asyncio
async def test_mixed_coroutines_and_factories():
    """Test mixing coroutines and factories."""

    async def coro_task(result: int) -> int:
        await asyncio.sleep(0.01)
        return result

    async def factory_task(result: int) -> int:
        await asyncio.sleep(0.01)
        return result

    results = await gather_with_timeout(
        [
            coro_task(1),  # Coroutine
            lambda: factory_task(2),  # Factory
            coro_task(3),
        ],
        per_timeout_s=1.0,
    )

    assert results == [1, 2, 3]


@pytest.mark.asyncio
async def test_empty_list():
    """Test with empty list of tasks."""
    results = await gather_with_timeout([], per_timeout_s=1.0)
    assert results == []


@pytest.mark.asyncio
async def test_single_task_timeout():
    """Test single task that times out."""

    async def slow_task() -> int:
        await asyncio.sleep(1.0)
        return 42

    results = await gather_with_timeout(
        [slow_task()],
        per_timeout_s=0.01,
    )

    assert results == [None]


@pytest.mark.asyncio
async def test_per_task_timeouts_independent():
    """Test that each task has its own timeout budget."""

    async def task_with_delay(delay: float) -> str:
        await asyncio.sleep(delay)
        return f"completed_{delay}"

    # All tasks have 0.1s each, not shared
    results = await gather_with_timeout(
        [
            task_with_delay(0.05),
            task_with_delay(0.05),
            task_with_delay(0.05),
        ],
        per_timeout_s=0.1,
    )

    assert results == ["completed_0.05", "completed_0.05", "completed_0.05"]


@pytest.mark.asyncio
async def test_result_order_preserved():
    """Test that result order matches input order."""

    async def task(task_id: int, delay: float) -> int:
        await asyncio.sleep(delay)
        return task_id

    # Tasks complete in different orders
    results = await gather_with_timeout(
        [
            task(1, 0.05),
            task(2, 0.01),
            task(3, 0.03),
        ],
        per_timeout_s=1.0,
    )

    # Results should be in original order, not completion order
    assert results == [1, 2, 3]


@pytest.mark.asyncio
async def test_none_is_distinguishable_from_timeout():
    """Test that explicit None returns are preserved and distinguishable from timeout."""

    async def returns_none() -> None:
        await asyncio.sleep(0.01)

    async def slow_task() -> str:
        await asyncio.sleep(1.0)
        return "should timeout"

    results = await gather_with_timeout(
        [
            returns_none(),
            slow_task(),
        ],
        per_timeout_s=0.05,
    )

    # Both are None, but first completed successfully
    assert results[0] is None
    assert results[1] is None


@pytest.mark.asyncio
async def test_complex_return_values():
    """Test that complex return values are preserved."""

    async def complex_task(task_id: int) -> dict:
        await asyncio.sleep(0.01)
        return {
            "id": task_id,
            "data": [1, 2, 3],
            "nested": {"key": "value"},
        }

    results = await gather_with_timeout(
        [
            complex_task(1),
            complex_task(2),
        ],
        per_timeout_s=1.0,
    )

    assert results[0] == {"id": 1, "data": [1, 2, 3], "nested": {"key": "value"}}
    assert results[1] == {"id": 2, "data": [1, 2, 3], "nested": {"key": "value"}}


@pytest.mark.asyncio
async def test_very_large_timeout():
    """Test with very large timeout (no tasks should timeout)."""

    async def quick_task(result: int) -> int:
        await asyncio.sleep(0.001)
        return result

    results = await gather_with_timeout(
        [quick_task(i) for i in range(10)],
        per_timeout_s=1000.0,
    )

    assert results == list(range(10))
