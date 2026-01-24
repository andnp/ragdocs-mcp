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
        deadline = asyncio.get_event_loop().time() + timeout
        poll_interval = 0.01

        while True:
            message = self.get_nowait()
            if message is not None:
                return message

            now = asyncio.get_event_loop().time()
            if now >= deadline:
                return None

            remaining = min(poll_interval, deadline - now)
            await asyncio.sleep(remaining)

    async def put(self, message: IPCMessage, timeout: float = 1.0) -> bool:
        deadline = asyncio.get_event_loop().time() + timeout
        poll_interval = 0.01

        while True:
            if self.put_nowait(message):
                return True

            now = asyncio.get_event_loop().time()
            if now >= deadline:
                return False

            remaining = min(poll_interval, deadline - now)
            await asyncio.sleep(remaining)

    def drain(self) -> list[IPCMessage]:
        messages: list[IPCMessage] = []
        while True:
            message = self.get_nowait()
            if message is None:
                break
            messages.append(message)
        return messages
