"""Unit tests for FileWatcher observer lifecycle and cleanup."""

from __future__ import annotations

import queue
from unittest.mock import MagicMock, patch

import pytest

from src.indexing.watcher import FileWatcher, _MarkdownEventHandler, MAX_QUEUE_SIZE


@pytest.fixture
def mock_index_manager():
    """Create a mock IndexManager."""
    manager = MagicMock()
    manager.index_document = MagicMock()
    manager.remove_document = MagicMock()
    manager.get_failed_files = MagicMock(return_value=[])
    return manager


@pytest.fixture
def watcher(tmp_path, mock_index_manager):
    """Create a FileWatcher instance for testing."""
    docs_path = tmp_path / "docs"
    docs_path.mkdir()
    return FileWatcher(
        documents_path=str(docs_path),
        index_manager=mock_index_manager,
        cooldown=0.1,
    )


@pytest.fixture
def mock_observer():
    """Create a mock observer for testing __del__ without event loop."""
    observer = MagicMock()
    observer.unschedule_all = MagicMock()
    observer.stop = MagicMock()
    observer.join = MagicMock()
    observer.is_alive = MagicMock(return_value=False)
    return observer


@pytest.fixture
def full_queue():
    """Create a full queue for testing drop behavior."""
    q = queue.Queue(maxsize=2)
    q.put_nowait(("created", "/path/a.md"))
    q.put_nowait(("created", "/path/b.md"))
    return q


@pytest.fixture
def event_handler(full_queue):
    """Create event handler with a full queue."""
    return _MarkdownEventHandler(full_queue)


class TestFileWatcherStoppedCleanly:
    """Tests for the stopped_cleanly property."""

    def test_stopped_cleanly_default_is_true(self, watcher):
        """stopped_cleanly should be True by default."""
        assert watcher.stopped_cleanly is True

    @pytest.mark.asyncio
    async def test_stopped_cleanly_after_normal_stop(self, watcher):
        """stopped_cleanly should remain True after normal stop."""
        watcher.start()
        await watcher.stop()
        assert watcher.stopped_cleanly is True

    @pytest.mark.asyncio
    async def test_stopped_cleanly_false_on_timeout(self, watcher):
        """stopped_cleanly should be False when observer times out."""
        watcher.start()

        # Mock the observer to hang on join
        def slow_join(timeout=None):
            # Simulate a stuck thread that doesn't join within timeout
            import time

            time.sleep(3.0)

        with patch.object(watcher._observer, "join", side_effect=slow_join):
            await watcher.stop()

        assert watcher.stopped_cleanly is False


class TestFileWatcherUnscheduleAll:
    """Tests for unschedule_all() behavior during stop."""

    @pytest.mark.asyncio
    async def test_unschedule_all_called_on_stop(self, watcher):
        """unschedule_all() should be called when stopping the watcher."""
        watcher.start()
        observer = watcher._observer

        with patch.object(observer, "unschedule_all") as mock_unschedule:
            # Also patch stop to prevent internal unschedule_all call
            with patch.object(observer, "stop"):
                await watcher.stop()
            # Called at least once during stop
            assert mock_unschedule.call_count >= 1

    @pytest.mark.asyncio
    async def test_unschedule_all_called_before_observer_stop(self, watcher):
        """unschedule_all() should be called before observer.stop()."""
        watcher.start()
        observer = watcher._observer
        call_order = []

        def track_unschedule():
            call_order.append("unschedule_all")

        def track_stop():
            call_order.append("stop")

        with (
            patch.object(observer, "unschedule_all", side_effect=track_unschedule),
            patch.object(observer, "stop", side_effect=track_stop),
        ):
            await watcher.stop()

        # First two calls should be unschedule_all then stop
        assert call_order[:2] == ["unschedule_all", "stop"]

    @pytest.mark.asyncio
    async def test_unschedule_all_exception_logged(self, watcher, caplog):
        """Exceptions from unschedule_all() should be logged but not raised."""
        watcher.start()
        observer = watcher._observer

        with (
            patch.object(
                observer, "unschedule_all", side_effect=RuntimeError("Watch error")
            ),
            patch.object(observer, "stop"),  # Prevent internal unschedule_all call
        ):
            # Should not raise
            await watcher.stop()

        assert "Failed to unschedule watches" in caplog.text
        assert "Watch error" in caplog.text


class TestFileWatcherTimeoutHandling:
    """Tests for observer thread timeout handling."""

    @pytest.mark.asyncio
    async def test_timeout_triggers_second_unschedule_attempt(self, watcher):
        """On timeout, a second unschedule_all() should be attempted."""
        watcher.start()
        observer = watcher._observer
        unschedule_call_count = 0

        def count_unschedule():
            nonlocal unschedule_call_count
            unschedule_call_count += 1

        def slow_join(timeout=None):
            import time

            time.sleep(3.0)

        with (
            patch.object(observer, "unschedule_all", side_effect=count_unschedule),
            patch.object(observer, "stop"),  # Prevent internal unschedule_all
            patch.object(observer, "join", side_effect=slow_join),
        ):
            await watcher.stop()

        # Should be called twice: once before stop, once on timeout
        assert unschedule_call_count == 2

    @pytest.mark.asyncio
    async def test_timeout_logs_warning(self, watcher, caplog):
        """Observer timeout should log a warning message."""
        watcher.start()

        def slow_join(timeout=None):
            import time

            time.sleep(3.0)

        with patch.object(watcher._observer, "join", side_effect=slow_join):
            await watcher.stop()

        assert "Observer thread did not stop within timeout" in caplog.text

    @pytest.mark.asyncio
    async def test_observer_set_to_none_after_timeout(self, watcher):
        """Observer should be set to None even after timeout."""
        watcher.start()

        def slow_join(timeout=None):
            import time

            time.sleep(3.0)

        with patch.object(watcher._observer, "join", side_effect=slow_join):
            await watcher.stop()

        assert watcher._observer is None


class TestFileWatcherDelMethod:
    """Tests for __del__ safety net method."""

    def test_del_calls_unschedule_all(self, watcher, mock_observer):
        """__del__ should attempt to unschedule all watches."""
        # Manually assign observer without starting (avoids event loop)
        watcher._observer = mock_observer
        watcher.__del__()
        mock_observer.unschedule_all.assert_called_once()

    def test_del_calls_stop(self, watcher, mock_observer):
        """__del__ should attempt to stop the observer."""
        watcher._observer = mock_observer
        watcher.__del__()
        mock_observer.stop.assert_called_once()

    def test_del_sets_observer_to_none(self, watcher, mock_observer):
        """__del__ should set observer to None."""
        watcher._observer = mock_observer
        watcher.__del__()
        assert watcher._observer is None

    def test_del_handles_no_observer(self, watcher):
        """__del__ should handle the case when observer is None."""
        # Don't start - observer is None
        watcher.__del__()  # Should not raise

    def test_del_suppresses_exceptions(self, watcher, mock_observer):
        """__del__ should suppress any exceptions."""
        mock_observer.unschedule_all.side_effect = RuntimeError("Error")
        watcher._observer = mock_observer
        # Should not raise
        watcher.__del__()


class TestFileWatcherStopIdempotent:
    """Tests for idempotent stop behavior."""

    @pytest.mark.asyncio
    async def test_stop_when_not_running(self, watcher):
        """Calling stop() when not running should be a no-op."""
        # Don't start
        await watcher.stop()  # Should not raise
        assert watcher.stopped_cleanly is True

    @pytest.mark.asyncio
    async def test_stop_twice_is_safe(self, watcher):
        """Calling stop() twice should be safe."""
        watcher.start()
        await watcher.stop()
        await watcher.stop()  # Second call should be no-op
        assert watcher._observer is None


class TestEventHandlerDroppedEventCounter:
    """Tests for _MarkdownEventHandler dropped event tracking."""

    def test_dropped_event_count_starts_at_zero(self):
        """Counter should start at zero."""
        q = queue.Queue(maxsize=10)
        handler = _MarkdownEventHandler(q)
        assert handler.dropped_event_count == 0

    def test_dropped_since_reconcile_starts_at_zero(self):
        """Per-reconcile counter should start at zero."""
        q = queue.Queue(maxsize=10)
        handler = _MarkdownEventHandler(q)
        assert handler.dropped_since_reconcile == 0

    def test_dropped_counter_increments_on_queue_full(self, event_handler):
        """Counter should increment when queue is full."""
        assert event_handler.dropped_event_count == 0
        event_handler._queue_event("created", "/path/new.md")
        assert event_handler.dropped_event_count == 1

    def test_dropped_since_reconcile_increments(self, event_handler):
        """Per-reconcile counter should increment when queue is full."""
        assert event_handler.dropped_since_reconcile == 0
        event_handler._queue_event("created", "/path/new.md")
        assert event_handler.dropped_since_reconcile == 1

    def test_both_counters_increment_together(self, event_handler):
        """Both counters should increment on each drop."""
        event_handler._queue_event("created", "/path/1.md")
        event_handler._queue_event("created", "/path/2.md")
        event_handler._queue_event("created", "/path/3.md")

        assert event_handler.dropped_event_count == 3
        assert event_handler.dropped_since_reconcile == 3

    def test_reset_dropped_counter_clears_reconcile_counter(self, event_handler):
        """reset_dropped_counter should clear only per-reconcile counter."""
        event_handler._queue_event("created", "/path/1.md")
        event_handler._queue_event("created", "/path/2.md")

        event_handler.reset_dropped_counter()

        assert event_handler.dropped_event_count == 2  # Total unchanged
        assert event_handler.dropped_since_reconcile == 0  # Reset

    def test_counter_increments_after_reset(self, event_handler):
        """Counter should continue incrementing after reset."""
        event_handler._queue_event("created", "/path/1.md")
        event_handler.reset_dropped_counter()
        event_handler._queue_event("created", "/path/2.md")

        assert event_handler.dropped_event_count == 2
        assert event_handler.dropped_since_reconcile == 1

    def test_log_throttling_at_100_drops(self, event_handler, caplog):
        """Warning should be logged every 100 drops."""
        # Fill queue and trigger drops
        for i in range(150):
            event_handler._queue_event("created", f"/path/{i}.md")

        # Should log at 100, not at 1, 50, 99, etc.
        log_messages = [r.message for r in caplog.records if r.levelname == "WARNING"]
        assert len(log_messages) == 1
        assert "dropped 100 total events" in log_messages[0]

    def test_log_message_includes_both_counters(self, event_handler, caplog):
        """Log message should include both total and per-reconcile counts."""
        # Drop 50, reset, drop 50 more to reach 100
        for i in range(50):
            event_handler._queue_event("created", f"/path/a{i}.md")
        event_handler.reset_dropped_counter()
        for i in range(50):
            event_handler._queue_event("created", f"/path/b{i}.md")

        log_messages = [r.message for r in caplog.records if r.levelname == "WARNING"]
        assert len(log_messages) == 1
        assert "100 total" in log_messages[0]
        assert "50 since last reconcile" in log_messages[0]


class TestFileWatcherDroppedEventMetrics:
    """Tests for FileWatcher dropped event metrics delegation."""

    def test_dropped_event_count_zero_when_not_started(self, watcher):
        """dropped_event_count should return 0 when watcher not started."""
        assert watcher.dropped_event_count == 0

    def test_dropped_since_reconcile_zero_when_not_started(self, watcher):
        """dropped_since_reconcile should return 0 when watcher not started."""
        assert watcher.dropped_since_reconcile == 0

    def test_should_reconcile_false_when_not_started(self, watcher):
        """should_reconcile should return False when watcher not started."""
        assert watcher.should_reconcile() is False

    def test_reset_dropped_counter_safe_when_not_started(self, watcher):
        """reset_dropped_counter should be safe when watcher not started."""
        watcher.reset_dropped_counter()  # Should not raise

    @pytest.mark.asyncio
    async def test_dropped_event_count_delegates_to_handler(self, watcher):
        """dropped_event_count should delegate to event handler."""
        watcher.start()
        # Access internal handler and force a drop
        handler = watcher._event_handler
        # Fill queue to capacity
        for _ in range(MAX_QUEUE_SIZE):
            try:
                watcher._event_queue.put_nowait(("created", "/tmp.md"))
            except queue.Full:
                break
        # Now force a drop via the handler
        handler._queue_event("created", "/path/dropped.md")
        assert watcher.dropped_event_count == 1
        await watcher.stop()

    @pytest.mark.asyncio
    async def test_should_reconcile_returns_true_after_drops(self, watcher):
        """should_reconcile should return True after events are dropped."""
        watcher.start()
        handler = watcher._event_handler
        # Fill queue
        for _ in range(MAX_QUEUE_SIZE):
            try:
                watcher._event_queue.put_nowait(("created", "/tmp.md"))
            except queue.Full:
                break
        # Force a drop
        handler._queue_event("created", "/path/dropped.md")
        assert watcher.should_reconcile() is True
        await watcher.stop()

    @pytest.mark.asyncio
    async def test_should_reconcile_false_after_reset(self, watcher):
        """should_reconcile should return False after reset."""
        watcher.start()
        handler = watcher._event_handler
        # Fill queue and force drop
        for _ in range(MAX_QUEUE_SIZE):
            try:
                watcher._event_queue.put_nowait(("created", "/tmp.md"))
            except queue.Full:
                break
        handler._queue_event("created", "/path/dropped.md")
        watcher.reset_dropped_counter()
        assert watcher.should_reconcile() is False
        await watcher.stop()

    @pytest.mark.asyncio
    async def test_reset_dropped_counter_delegates_to_handler(self, watcher):
        """reset_dropped_counter should delegate to event handler."""
        watcher.start()
        handler = watcher._event_handler
        # Fill queue and force drop
        for _ in range(MAX_QUEUE_SIZE):
            try:
                watcher._event_queue.put_nowait(("created", "/tmp.md"))
            except queue.Full:
                break
        handler._queue_event("created", "/path/dropped.md")
        watcher.reset_dropped_counter()
        assert handler.dropped_since_reconcile == 0
        await watcher.stop()
