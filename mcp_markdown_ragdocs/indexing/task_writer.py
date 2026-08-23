"""Writer lease coordination for indexing tasks."""

from __future__ import annotations

import functools
import logging
import sys
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any
from uuid import uuid4

from mcp_markdown_ragdocs.coordination.task_leases import TaskLeasePort, TaskLeaseStore

logger = logging.getLogger(__name__)

WRITER_LEASE_TIMEOUT_SECONDS = 30.0
WRITER_HEARTBEAT_INTERVAL_SECONDS = 5.0
INDEX_WRITER_RESOURCE = "index-writer"


def writer_lease_store(queue_path: str | Path) -> TaskLeaseStore:
    return TaskLeaseStore(Path(queue_path), timeout_seconds=WRITER_LEASE_TIMEOUT_SECONDS)


def writer_is_active(
    store_factory: Callable[[], TaskLeasePort | None],
) -> bool:
    store = store_factory()
    return store is not None and store.writer_owner(INDEX_WRITER_RESOURCE) is not None


def run_as_writer(
    store_factory: Callable[[], TaskLeasePort | None],
    operation: Callable[[], Any],
    *,
    operation_name: str = "index write",
    operation_args: tuple[Any, ...] = (),
    owner_token: str | None = None,
    busy_result: Any,
    on_busy: Callable[[], None] | None = None,
    on_released: Callable[[], None] | None = None,
) -> Any:
    store = store_factory()
    if store is None:
        return busy_result

    token = owner_token or uuid4().hex
    if not store.acquire_writer(INDEX_WRITER_RESOURCE, token):
        details = ""
        if operation_args and isinstance(operation_args[0], str):
            details = f" argument={operation_args[0]!r}"
        elif operation_args and isinstance(operation_args[0], list):
            batch = operation_args[0]
            details = f" batch_size={len(batch)}"
            if all(isinstance(item, str) for item in batch):
                details += f" batch_preview={batch[:3]!r}"
        logger.warning("Writer lease busy; deferring %s%s", operation_name, details)
        if on_busy is not None:
            on_busy()
        return busy_result

    heartbeat_stop = threading.Event()

    def _heartbeat() -> None:
        while not heartbeat_stop.wait(WRITER_HEARTBEAT_INTERVAL_SECONDS):
            if not store.heartbeat_writer(INDEX_WRITER_RESOURCE, token):
                logger.warning("Lost index writer ownership: %s", token)
                return

    heartbeat_thread = threading.Thread(
        target=_heartbeat,
        name=f"index-writer-heartbeat-{token[:8]}",
        daemon=True,
    )
    heartbeat_thread.start()
    try:
        return operation()
    finally:
        heartbeat_stop.set()
        heartbeat_thread.join(timeout=WRITER_HEARTBEAT_INTERVAL_SECONDS)
        released = store.release_writer(INDEX_WRITER_RESOURCE, token)
        if released and on_released is not None:
            operation_error = sys.exc_info()[1]
            try:
                on_released()
            except Exception:
                if operation_error is None:
                    raise
                logger.exception("Writer release callback failed after operation error")


def writer_owned_task(
    store_factory: Callable[[], TaskLeasePort | None],
    *,
    operation: str | None = None,
    busy_result: Any,
    on_busy: Callable[..., None] | None = None,
    on_released: Callable[[], None] | None = None,
):
    def _decorate(function):
        @functools.wraps(function)
        def _wrapped(*args, **kwargs):
            busy_callback = on_busy
            return run_as_writer(
                store_factory,
                lambda: function(*args, **kwargs),
                operation_name=operation or function.__name__,
                operation_args=args,
                busy_result=busy_result,
                on_busy=(
                    None
                    if busy_callback is None
                    else lambda: busy_callback(*args, **kwargs)
                ),
                on_released=on_released,
            )

        return _wrapped

    return _decorate
