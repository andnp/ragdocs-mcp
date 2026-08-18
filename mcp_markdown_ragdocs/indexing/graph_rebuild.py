"""Asynchronous, coalescing graph rebuild coordination."""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable

logger = logging.getLogger(__name__)


class DebouncedGraphRebuilder:
    """Run one rebuild after a quiet period, coalescing requests made during it.

    The rebuild carries no payload: requests made while one is queued or running
    collapse into a single later run, so the callable must read whatever work
    accumulated for itself rather than receive a snapshot that a newer request
    would silently replace.
    """

    def __init__(
        self,
        rebuild: Callable[[], None],
        *,
        debounce_seconds: float,
    ) -> None:
        if debounce_seconds < 0:
            raise ValueError("debounce_seconds must be non-negative")
        self._rebuild = rebuild
        self._debounce_seconds = debounce_seconds
        self._condition = threading.Condition()
        self._pending = False
        self._due_at: float | None = None
        self._running = False
        self._closed = False
        self._error: BaseException | None = None
        self._thread = threading.Thread(
            target=self._run,
            name="ragdocs-graph-rebuild",
            daemon=True,
        )
        self._thread.start()

    def request(self) -> None:
        with self._condition:
            if self._closed:
                return
            self._pending = True
            self._due_at = time.monotonic() + self._debounce_seconds
            self._condition.notify_all()

    def flush(self) -> None:
        with self._condition:
            if self._closed:
                return
            if self._pending:
                self._due_at = time.monotonic()
                self._condition.notify_all()
            while self._pending or self._running:
                self._condition.wait()
            error = self._error
            self._error = None
        if error is not None:
            raise error

    def close(self) -> None:
        with self._condition:
            if self._closed:
                return
            self._closed = True
            self._due_at = time.monotonic()
            self._condition.notify_all()
            while self._pending or self._running:
                self._condition.wait()
            error = self._error
            self._error = None
        self._thread.join()
        if error is not None:
            raise error

    def _run(self) -> None:
        while True:
            with self._condition:
                while not self._pending and not self._closed:
                    self._condition.wait()
                if not self._pending and self._closed:
                    return
                if not self._closed and self._due_at is not None:
                    remaining = self._due_at - time.monotonic()
                    if remaining > 0:
                        self._condition.wait(timeout=remaining)
                        continue
                self._pending = False
                self._due_at = None
                self._running = True

            try:
                self._rebuild()
            except BaseException as error:
                logger.exception("Asynchronous graph rebuild failed")
                with self._condition:
                    self._error = error
            finally:
                with self._condition:
                    self._running = False
                    self._condition.notify_all()
