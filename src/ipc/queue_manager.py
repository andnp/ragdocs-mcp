from __future__ import annotations

import asyncio
import logging
import multiprocessing
from queue import Empty, Full

from src.ipc.commands import IPCMessage

logger = logging.getLogger(__name__)


class QueueManager:
    def __init__(self, queue: multiprocessing.Queue[IPCMessage], name: str = "queue"):
        self._queue = queue
        self._name = name

    def put_nowait(self, message: IPCMessage) -> bool:
        try:
            self._queue.put_nowait(message)
            return True
        except Full:
            logger.warning("Queue %s is full, dropping message: %s", self._name, type(message).__name__)
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
