"""
Unit tests for QueueManager async wrapper.

Tests async get/put operations, timeouts, and drain functionality.
"""

import asyncio
import threading
from multiprocessing import Queue

import pytest

from src.ipc.commands import HealthCheckCommand, ShutdownCommand
from src.ipc.queue_manager import CommandPriority, QueueManager


@pytest.fixture
def mp_queue() -> Queue:
    """Create a multiprocessing Queue for testing."""
    return Queue()


@pytest.fixture
def small_queue() -> Queue:
    """Create a small multiprocessing Queue that fills quickly."""
    return Queue(maxsize=2)


@pytest.fixture
def queue_manager(mp_queue: Queue) -> QueueManager:
    """Create a QueueManager wrapping the test queue."""
    return QueueManager(mp_queue, name="test_queue")


@pytest.fixture
def small_queue_manager(small_queue: Queue) -> QueueManager:
    """Create a QueueManager with a small underlying queue."""
    return QueueManager(small_queue, name="small_test_queue")


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


class TestCommandPriority:
    """Tests for CommandPriority enum."""

    def test_priority_ordering(self):
        """Verify priority levels have correct ordering."""
        assert CommandPriority.CRITICAL < CommandPriority.NORMAL < CommandPriority.LOW

    def test_priority_values(self):
        """Verify priority enum values."""
        assert CommandPriority.CRITICAL == 0
        assert CommandPriority.NORMAL == 1
        assert CommandPriority.LOW == 2


class TestQueueManagerDroppedCount:
    """Tests for dropped message tracking."""

    def test_initial_dropped_count_is_zero(self, queue_manager: QueueManager):
        """Verify dropped count starts at zero."""
        assert queue_manager.get_dropped_count() == 0

    def test_dropped_count_increments_on_full_queue(
        self, small_queue_manager: QueueManager, small_queue: Queue
    ):
        """Verify dropped count increments when queue is full."""
        # Fill the queue (maxsize=2)
        small_queue_manager.put_nowait(ShutdownCommand())
        small_queue_manager.put_nowait(ShutdownCommand())

        assert small_queue_manager.get_dropped_count() == 0

        # This should fail and increment dropped count
        result = small_queue_manager.put_nowait(ShutdownCommand())

        assert result is False
        assert small_queue_manager.get_dropped_count() == 1

    def test_dropped_count_accumulates(
        self, small_queue_manager: QueueManager, small_queue: Queue
    ):
        """Verify multiple dropped messages accumulate count."""
        # Fill the queue
        small_queue_manager.put_nowait(ShutdownCommand())
        small_queue_manager.put_nowait(ShutdownCommand())

        # Drop several messages
        for _ in range(5):
            small_queue_manager.put_nowait(ShutdownCommand())

        assert small_queue_manager.get_dropped_count() == 5

    def test_dropped_count_thread_safety(self, small_queue_manager: QueueManager):
        """Verify dropped count is thread-safe under concurrent access."""
        # Fill the queue first
        small_queue_manager.put_nowait(ShutdownCommand())
        small_queue_manager.put_nowait(ShutdownCommand())

        threads_count = 10
        drops_per_thread = 100
        errors: list[Exception] = []

        def drop_messages():
            try:
                for _ in range(drops_per_thread):
                    small_queue_manager.put_nowait(ShutdownCommand())
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=drop_messages) for _ in range(threads_count)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert small_queue_manager.get_dropped_count() == threads_count * drops_per_thread


class TestQueueManagerPutCritical:
    """Tests for put_critical() blocking method."""

    def test_put_critical_success(self, queue_manager: QueueManager, mp_queue: Queue):
        """Verify critical messages are delivered successfully."""
        result = queue_manager.put_critical(ShutdownCommand(), timeout=1.0)

        assert result is True
        msg = mp_queue.get(timeout=1.0)
        assert isinstance(msg, ShutdownCommand)

    def test_put_critical_blocks_until_space(self, small_queue: Queue):
        """Verify put_critical blocks when queue is full and succeeds when space opens."""
        qm = QueueManager(small_queue, name="critical_test")

        # Fill the queue
        qm.put_nowait(ShutdownCommand())
        qm.put_nowait(ShutdownCommand())

        result_holder: list[bool] = []

        def producer():
            # This should block until space is available
            result = qm.put_critical(HealthCheckCommand(), timeout=2.0)
            result_holder.append(result)

        def consumer():
            import time
            time.sleep(0.1)  # Let producer start blocking
            small_queue.get()  # Free up one slot

        producer_thread = threading.Thread(target=producer)
        consumer_thread = threading.Thread(target=consumer)

        producer_thread.start()
        consumer_thread.start()

        producer_thread.join(timeout=3.0)
        consumer_thread.join(timeout=1.0)

        assert len(result_holder) == 1
        assert result_holder[0] is True

    def test_put_critical_timeout_on_full_queue(self, small_queue: Queue):
        """Verify put_critical returns False after timeout on full queue."""
        qm = QueueManager(small_queue, name="timeout_test")

        # Fill the queue
        qm.put_nowait(ShutdownCommand())
        qm.put_nowait(ShutdownCommand())

        # This should timeout and return False
        result = qm.put_critical(HealthCheckCommand(), timeout=0.1)

        assert result is False


class TestQueueManagerPutCriticalAsync:
    """Tests for put_critical_async() method."""

    @pytest.mark.asyncio
    async def test_put_critical_async_success(
        self, queue_manager: QueueManager, mp_queue: Queue
    ):
        """Verify async critical put succeeds."""
        result = await queue_manager.put_critical_async(ShutdownCommand(), timeout=1.0)

        assert result is True
        msg = mp_queue.get(timeout=1.0)
        assert isinstance(msg, ShutdownCommand)

    @pytest.mark.asyncio
    async def test_put_critical_async_timeout(self, small_queue: Queue):
        """Verify async critical put times out on full queue."""
        qm = QueueManager(small_queue, name="async_timeout_test")

        # Fill the queue
        qm.put_nowait(ShutdownCommand())
        qm.put_nowait(ShutdownCommand())

        result = await qm.put_critical_async(HealthCheckCommand(), timeout=0.1)

        assert result is False

    @pytest.mark.asyncio
    async def test_put_critical_async_blocks_until_space(self, small_queue: Queue):
        """Verify async put_critical blocks and succeeds when space opens."""
        qm = QueueManager(small_queue, name="async_block_test")

        # Fill the queue
        qm.put_nowait(ShutdownCommand())
        qm.put_nowait(ShutdownCommand())

        async def consumer():
            await asyncio.sleep(0.1)
            small_queue.get()  # Free up space

        # Run producer and consumer concurrently
        producer_task = asyncio.create_task(
            qm.put_critical_async(HealthCheckCommand(), timeout=2.0)
        )
        consumer_task = asyncio.create_task(consumer())

        result = await producer_task
        await consumer_task

        assert result is True
