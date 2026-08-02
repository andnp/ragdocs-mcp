"""Huey consumer wrapper for running task workers."""

from __future__ import annotations

import logging
import threading
import uuid
from typing import TYPE_CHECKING, Any, cast

from huey.constants import EmptyData
from huey.utils import Error

from mcp_markdown_ragdocs.coordination.task_leases import TaskLeaseStore

if TYPE_CHECKING:
    from huey import SqliteHuey

logger = logging.getLogger(__name__)

LEASE_TIMEOUT_SECONDS = 30.0
LEASE_HEARTBEAT_INTERVAL_SECONDS = 5.0


class HueyWorker:
    """Manages a Huey consumer thread for processing background tasks.

    Runs in the same process as the main server, in a daemon thread.
    Only started when the lifecycle state is READY_PRIMARY.
    """

    def __init__(self, huey: SqliteHuey, workers: int = 2) -> None:
        self._huey = huey
        self._workers = workers
        self._consumer: _HueyConsumerThread | None = None
        self._started = False

    @property
    def is_running(self) -> bool:
        return (
            self._started and self._consumer is not None and self._consumer.is_alive()
        )

    def start(self) -> None:
        """Start the consumer thread."""
        if self._started:
            logger.warning("HueyWorker already started")
            return

        lease_store = TaskLeaseStore(
            cast(Any, self._huey.storage).filename,
            timeout_seconds=LEASE_TIMEOUT_SECONDS,
        )
        for lease in lease_store.reclaim_expired():
            if lease.payload is None:
                raise RuntimeError(
                    f"Expired lease {lease.task_id} has no serialized task payload"
                )
            self._huey.enqueue(self._huey.deserialize_task(lease.payload))

        self._consumer = _HueyConsumerThread(
            self._huey,
            self._workers,
            lease_store=lease_store,
        )
        self._consumer.start()
        self._started = True
        logger.info("Huey worker started with %d workers", self._workers)

    def stop(self, timeout: float = 5.0) -> None:
        """Stop the consumer thread."""
        if not self._started or self._consumer is None:
            return

        self._consumer.request_stop()
        self._consumer.join(timeout=timeout)

        if self._consumer.is_alive():
            logger.warning("Huey consumer thread did not stop within %.1fs", timeout)
        else:
            logger.info("Huey worker stopped")

        self._consumer = None
        self._started = False


class _HueyConsumerThread(threading.Thread):
    """Thread running the Huey consumer loop."""

    def __init__(
        self,
        huey: SqliteHuey,
        workers: int,
        *,
        lease_store: TaskLeaseStore,
    ) -> None:
        super().__init__(name="huey-consumer", daemon=True)
        self._huey = huey
        self._workers = workers
        self._lease_store = lease_store
        self._owner_token = uuid.uuid4().hex
        self._stop_event = threading.Event()

    def request_stop(self) -> None:
        self._stop_event.set()

    def run(self) -> None:
        """Run the consumer loop, processing tasks from the queue."""
        logger.info("Huey consumer thread started")
        try:
            while not self._stop_event.is_set():
                # Dequeue and execute one task at a time
                task = self._huey.dequeue()
                if task is not None:
                    self._execute_task(task)
                else:
                    # No task available, wait briefly
                    self._stop_event.wait(timeout=0.5)
        except Exception:
            logger.exception("Huey consumer thread error")
        finally:
            logger.info("Huey consumer thread exiting")

    def _execute_task(self, task: Any) -> None:
        task_id = str(task.id)
        task_name = getattr(task, "name", None)
        claimed = self._lease_store.claim(
            task_id,
            task_name=task_name if isinstance(task_name, str) else None,
            owner_token=self._owner_token,
            payload=self._huey.serialize_task(task),
        )
        if not claimed:
            logger.warning("Skipping task with an active lease: %s", task_id)
            return

        heartbeat_stop = threading.Event()
        heartbeat_thread = threading.Thread(
            target=self._heartbeat,
            args=(task_id, heartbeat_stop),
            name=f"huey-heartbeat-{task_id[:8]}",
            daemon=True,
        )
        heartbeat_thread.start()
        try:
            self._huey.execute(task)
        except Exception as exc:
            self._lease_store.fail(
                task_id,
                owner_token=self._owner_token,
                error=f"{type(exc).__name__}: {exc}",
            )
            logger.exception("Task execution failed")
        else:
            execution_error = self._read_execution_error(task)
            if execution_error is None:
                self._lease_store.complete(
                    task_id,
                    owner_token=self._owner_token,
                )
            else:
                self._lease_store.fail(
                    task_id,
                    owner_token=self._owner_token,
                    error=execution_error,
                )
        finally:
            heartbeat_stop.set()
            heartbeat_thread.join(timeout=LEASE_HEARTBEAT_INTERVAL_SECONDS)

    def _heartbeat(self, task_id: str, stop_event: threading.Event) -> None:
        while not stop_event.wait(LEASE_HEARTBEAT_INTERVAL_SECONDS):
            if not self._lease_store.heartbeat(
                task_id,
                owner_token=self._owner_token,
            ):
                logger.warning("Lost task lease ownership: %s", task_id)
                return

    def _read_execution_error(self, task: Any) -> str | None:
        raw_item = cast(Any, self._huey.storage).peek_data(task.id)
        if raw_item is None or raw_item is EmptyData:
            return None
        payload = self._huey.serializer.deserialize(raw_item)
        if not isinstance(payload, Error):
            return None
        metadata = payload.metadata if isinstance(payload.metadata, dict) else {}
        return str(metadata.get("error", "task execution failed"))
