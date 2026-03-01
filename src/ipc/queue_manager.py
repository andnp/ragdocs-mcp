from __future__ import annotations

import asyncio
import logging
import multiprocessing
import threading
from enum import IntEnum
from queue import Empty, Full

from src.ipc.commands import IPCMessage

logger = logging.getLogger(__name__)


class CommandPriority(IntEnum):
    CRITICAL = 0  # Shutdown, health check
    NORMAL = 1  # Index updates, reindex
    LOW = 2  # Status queries


class QueueManager:
    def __init__(self, queue: multiprocessing.Queue[IPCMessage], name: str = "queue"):
        self._queue = queue
        self._name = name
        self._dropped_count = 0
        self._dropped_lock = threading.Lock()

    def put_nowait(self, message: IPCMessage) -> bool:
        try:
            self._queue.put_nowait(message)
            return True
        except Full:
            with self._dropped_lock:
                self._dropped_count += 1
            logger.warning("Queue %s is full, dropping message: %s", self._name, type(message).__name__)
            return False

    def get_dropped_count(self) -> int:
        with self._dropped_lock:
            return self._dropped_count

    def put_critical(self, message: IPCMessage, timeout: float = 5.0) -> bool:
        """Put a critical message using blocking put with timeout.

        Critical commands (shutdown, health check) should not be dropped.
        This method blocks up to `timeout` seconds to ensure delivery.
        """
        try:
            self._queue.put(message, block=True, timeout=timeout)
            return True
        except Full:
            logger.error(
                "Queue %s is full after %ss, critical message lost: %s",
                self._name,
                timeout,
                type(message).__name__,
            )
            return False

    async def put_critical_async(self, message: IPCMessage, timeout: float = 5.0) -> bool:
        """Async version of put_critical for use in async contexts."""
        try:
            await asyncio.to_thread(self._queue.put, message, True, timeout)
            return True
        except Full:
            logger.error(
                "Queue %s is full after %ss, critical message lost: %s",
                self._name,
                timeout,
                type(message).__name__,
            )
            return False

    def get_nowait(self) -> IPCMessage | None:
        try:
            return self._queue.get_nowait()
        except Empty:
            return None

    async def get(self, timeout: float = 1.0) -> IPCMessage | None:
        """Get a message from the queue, blocking up to timeout seconds."""
        try:
            return await asyncio.to_thread(self._queue.get, True, timeout)
        except Empty:
            return None

    async def put(self, message: IPCMessage, timeout: float = 1.0) -> bool:
        """Put a message on the queue, blocking up to timeout seconds."""
        try:
            await asyncio.to_thread(self._queue.put, message, True, timeout)
            return True
        except Full:
            logger.warning("Queue %s is full after %ss", self._name, timeout)
            return False

    def drain(self) -> list[IPCMessage]:
        messages: list[IPCMessage] = []
        while True:
            message = self.get_nowait()
            if message is None:
                break
            messages.append(message)
        return messages
