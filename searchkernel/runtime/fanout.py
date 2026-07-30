"""Async per-source-timeout fan-out helper."""

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


async def gather_with_timeout[T](
    coros_or_factories: list[Awaitable[T] | Callable[[], Awaitable[T]]],
    per_timeout_s: float,
) -> list[T | None]:
    """Run async tasks concurrently with per-task timeout.

    Each task is bounded by per_timeout_s. A task that times out or raises
    an exception yields None in its slot without failing the entire gather.

    Args:
        coros_or_factories: List of coroutines or callables that return coroutines.
                           Callables are invoked to create coroutines (lazy creation).
        per_timeout_s: Per-task timeout in seconds.

    Returns:
        List of results in the same order as input. Timed-out or failed tasks
        yield None.

    Example:
        async def fetch_source(name: str) -> dict:
            ...

        results = await gather_with_timeout(
            [
                fetch_source("source1"),
                fetch_source("source2"),
            ],
            per_timeout_s=5.0,
        )
        # results might be [{"data": ...}, None] if source2 timed out
    """
    tasks: list[asyncio.Task[T | None]] = []

    for coro_or_factory in coros_or_factories:
        # If it's a callable (factory), invoke it to create the coroutine
        if callable(coro_or_factory):
            coro = coro_or_factory()
        else:
            coro = coro_or_factory

        # Wrap the coroutine with timeout handling
        async def run_with_timeout(
            c: Awaitable[T], timeout: float
        ) -> T | None:
            try:
                return await asyncio.wait_for(c, timeout=timeout)
            except TimeoutError:
                logger.debug(
                    f"Task timed out after {timeout}s"
                )
                return None
            except Exception as e:  # noqa: BLE001 -- generic fan-out executor over heterogeneous tasks
                logger.debug(
                    f"Task failed with exception: {type(e).__name__}: {e}"
                )
                return None

        task = asyncio.create_task(run_with_timeout(coro, per_timeout_s))
        tasks.append(task)

    # Run all tasks and collect results
    results = await asyncio.gather(*tasks, return_exceptions=False)
    return results
