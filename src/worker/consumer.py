"""Huey consumer wrapper for running task workers."""

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from huey import SqliteHuey

logger = logging.getLogger(__name__)


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

        self._consumer = _HueyConsumerThread(self._huey, self._workers)
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

    def __init__(self, huey: SqliteHuey, workers: int) -> None:
        super().__init__(name="huey-consumer", daemon=True)
        self._huey = huey
        self._workers = workers
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
                    try:
                        self._huey.execute(task)
                    except Exception:
                        logger.error("Task execution failed", exc_info=True)
                else:
                    # No task available, wait briefly
                    self._stop_event.wait(timeout=0.5)
        except Exception:
            logger.error("Huey consumer thread error", exc_info=True)
        finally:
            logger.info("Huey consumer thread exiting")
