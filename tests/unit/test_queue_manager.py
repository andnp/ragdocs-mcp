"""
Unit tests for QueueManager async wrapper.

Tests async get/put operations, timeouts, and drain functionality.
"""

import asyncio
from multiprocessing import Queue

import pytest

from src.ipc.commands import HealthCheckCommand, ShutdownCommand
from src.ipc.queue_manager import QueueManager


@pytest.fixture
def mp_queue() -> Queue:
    """Create a multiprocessing Queue for testing."""
    return Queue()


@pytest.fixture
def queue_manager(mp_queue: Queue) -> QueueManager:
    """Create a QueueManager wrapping the test queue."""
    return QueueManager(mp_queue, name="test_queue")


class TestQueueManagerPut:
    """Tests for QueueManager.put() async method."""

    @pytest.mark.asyncio
    async def test_put_message(self, queue_manager: QueueManager, mp_queue: Queue):
        """Verify messages are put into the underlying queue."""
        cmd = ShutdownCommand()
        await queue_manager.put(cmd)

        # Verify message arrived in underlying queue
        result = mp_queue.get(timeout=1.0)
        assert isinstance(result, ShutdownCommand)
        assert result.graceful is True

    @pytest.mark.asyncio
    async def test_put_multiple_messages(self, queue_manager: QueueManager, mp_queue: Queue):
        """Verify multiple messages maintain order."""
        await queue_manager.put(ShutdownCommand(graceful=True))
        await queue_manager.put(ShutdownCommand(graceful=False))

        msg1 = mp_queue.get(timeout=1.0)
        msg2 = mp_queue.get(timeout=1.0)

        assert msg1.graceful is True
        assert msg2.graceful is False


class TestQueueManagerGet:
    """Tests for QueueManager.get() async method."""

    @pytest.mark.asyncio
    async def test_get_message(self, queue_manager: QueueManager, mp_queue: Queue):
        """Verify messages can be retrieved asynchronously."""
        cmd = HealthCheckCommand()
        mp_queue.put(cmd)

        result = await queue_manager.get(timeout=1.0)
        assert isinstance(result, HealthCheckCommand)

    @pytest.mark.asyncio
    async def test_get_timeout(self, queue_manager: QueueManager):
        """Verify get returns None on timeout when queue is empty."""
        result = await queue_manager.get(timeout=0.1)
        assert result is None


class TestQueueManagerPutNowait:
    """Tests for QueueManager.put_nowait() synchronous method."""

    def test_put_nowait_success(self, queue_manager: QueueManager, mp_queue: Queue):
        """Verify put_nowait returns True on success."""
        result = queue_manager.put_nowait(ShutdownCommand())
        assert result is True

        # Verify message was queued
        msg = mp_queue.get(timeout=1.0)
        assert isinstance(msg, ShutdownCommand)


class TestQueueManagerGetNowait:
    """Tests for QueueManager.get_nowait() synchronous method."""

    def test_get_nowait_with_message(self, queue_manager: QueueManager, mp_queue: Queue):
        """Verify get_nowait returns message if available."""
        import time
        mp_queue.put(HealthCheckCommand())
        time.sleep(0.01)  # Allow message to propagate across process boundary

        result = queue_manager.get_nowait()
        assert isinstance(result, HealthCheckCommand)

    def test_get_nowait_empty_queue(self, queue_manager: QueueManager):
        """Verify get_nowait returns None on empty queue."""
        result = queue_manager.get_nowait()
        assert result is None


class TestQueueManagerDrain:
    """Tests for QueueManager.drain() method."""

    def test_drain_empty_queue(self, queue_manager: QueueManager):
        """Verify drain returns empty list for empty queue."""
        result = queue_manager.drain()
        assert result == []

    def test_drain_with_messages(self, queue_manager: QueueManager, mp_queue: Queue):
        """Verify drain returns all queued messages."""
        import time
        mp_queue.put(ShutdownCommand())
        mp_queue.put(HealthCheckCommand())
        mp_queue.put(ShutdownCommand(graceful=False))
        time.sleep(0.01)  # Allow messages to propagate

        result = queue_manager.drain()

        assert len(result) == 3
        assert isinstance(result[0], ShutdownCommand)
        assert isinstance(result[1], HealthCheckCommand)
        assert isinstance(result[2], ShutdownCommand)
        assert result[2].graceful is False

    def test_drain_empties_queue(self, queue_manager: QueueManager, mp_queue: Queue):
        """Verify drain removes all messages from queue."""
        import time
        mp_queue.put(ShutdownCommand())
        mp_queue.put(HealthCheckCommand())
        time.sleep(0.01)  # Allow messages to propagate

        queue_manager.drain()

        # Queue should now be empty
        result = queue_manager.get_nowait()
        assert result is None


class TestQueueManagerConcurrency:
    """Tests for concurrent queue operations."""

    @pytest.mark.asyncio
    async def test_concurrent_put_get(self, queue_manager: QueueManager):
        """
        Verify concurrent put and get operations don't deadlock.

        This test ensures the async wrapper handles concurrent access
        without issues.
        """
        async def producer():
            for i in range(10):
                await queue_manager.put(ShutdownCommand(timeout=float(i)))
                await asyncio.sleep(0.01)

        async def consumer():
            received = []
            for _ in range(10):
                msg = await queue_manager.get(timeout=1.0)
                if msg:
                    received.append(msg)
                await asyncio.sleep(0.005)
            return received

        producer_task = asyncio.create_task(producer())
        consumer_task = asyncio.create_task(consumer())

        await producer_task
        received = await consumer_task

        # Should receive all 10 messages
        assert len(received) == 10
