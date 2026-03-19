"""Unit tests for FileWatcher observer lifecycle and cleanup."""

from __future__ import annotations

import asyncio
import queue
from unittest.mock import MagicMock, patch

import pytest

from src.indexing.discovery import is_excluded_dir, walk_included_dirs
from src.indexing.tasks import TaskSubmissionResult
from src.indexing.watcher import (
    FileWatcher,
    _DocumentEventHandler,
    MAX_QUEUE_SIZE,
)


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
    return _DocumentEventHandler(full_queue, {".md", ".markdown"})


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
    """Tests for _DocumentEventHandler dropped event tracking."""

    def test_dropped_event_count_starts_at_zero(self):
        """Counter should start at zero."""
        q = queue.Queue(maxsize=10)
        handler = _DocumentEventHandler(q, {".md", ".markdown"})
        assert handler.dropped_event_count == 0

    def test_dropped_since_reconcile_starts_at_zero(self):
        """Per-reconcile counter should start at zero."""
        q = queue.Queue(maxsize=10)
        handler = _DocumentEventHandler(q, {".md", ".markdown"})
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


class TestDocumentEventHandlerSuffixFiltering:
    """Tests for _DocumentEventHandler suffix-based file filtering."""

    def test_supports_md_by_default(self):
        """Handler with default suffixes should accept .md files."""
        q = queue.Queue(maxsize=10)
        handler = _DocumentEventHandler(q, {".md", ".markdown"})
        assert handler._is_supported_file("/docs/readme.md") is True

    def test_supports_markdown_by_default(self):
        """Handler with default suffixes should accept .markdown files."""
        q = queue.Queue(maxsize=10)
        handler = _DocumentEventHandler(q, {".md", ".markdown"})
        assert handler._is_supported_file("/docs/readme.markdown") is True

    def test_rejects_unsupported_extension(self):
        """Handler should reject files with unsupported extensions."""
        q = queue.Queue(maxsize=10)
        handler = _DocumentEventHandler(q, {".md"})
        assert handler._is_supported_file("/docs/readme.txt") is False

    def test_supports_txt_when_configured(self):
        """Handler with .txt suffix should accept .txt files."""
        q = queue.Queue(maxsize=10)
        handler = _DocumentEventHandler(q, {".md", ".txt"})
        assert handler._is_supported_file("/docs/notes.txt") is True

    def test_supports_custom_extension(self):
        """Handler should accept files matching custom suffixes."""
        q = queue.Queue(maxsize=10)
        handler = _DocumentEventHandler(q, {".rst"})
        assert handler._is_supported_file("/docs/guide.rst") is True
        assert handler._is_supported_file("/docs/guide.md") is False

    def test_case_insensitive_suffix_matching(self):
        """Suffix matching should be case-insensitive."""
        q = queue.Queue(maxsize=10)
        handler = _DocumentEventHandler(q, {".md"})
        assert handler._is_supported_file("/docs/README.MD") is True

    def test_handles_bytes_path(self):
        """Handler should handle bytes paths correctly."""
        q = queue.Queue(maxsize=10)
        handler = _DocumentEventHandler(q, {".md", ".txt"})
        assert handler._is_supported_file(b"/docs/readme.md") is True
        assert handler._is_supported_file(b"/docs/notes.txt") is True
        assert handler._is_supported_file(b"/docs/script.py") is False


class TestFileWatcherParserSuffixes:
    """Tests for FileWatcher parser_suffixes parameter."""

    def test_default_suffixes_when_none(self, tmp_path, mock_index_manager):
        """FileWatcher should use default suffixes when None is passed."""
        docs_path = tmp_path / "docs"
        docs_path.mkdir()
        watcher = FileWatcher(
            documents_path=str(docs_path),
            index_manager=mock_index_manager,
            parser_suffixes=None,
        )
        assert watcher._parser_suffixes == {".md", ".markdown", ".txt"}

    def test_custom_suffixes_passed_through(self, tmp_path, mock_index_manager):
        """FileWatcher should store custom parser_suffixes."""
        docs_path = tmp_path / "docs"
        docs_path.mkdir()
        watcher = FileWatcher(
            documents_path=str(docs_path),
            index_manager=mock_index_manager,
            parser_suffixes={".md", ".txt", ".rst"},
        )
        assert watcher._parser_suffixes == {".md", ".txt", ".rst"}


class TestFileWatcherTaskMode:
    @pytest.mark.asyncio
    async def test_batch_process_accepts_hidden_root_but_rejects_hidden_child(
        self, tmp_path, mock_index_manager
    ):
        docs_path = tmp_path / ".hidden-root" / "docs"
        docs_path.mkdir(parents=True)
        visible_file = docs_path / "note.md"
        hidden_file = docs_path / ".hidden" / "secret.md"
        hidden_file.parent.mkdir()

        watcher = FileWatcher(
            documents_path=str(docs_path),
            index_manager=mock_index_manager,
        )

        await watcher._batch_process(
            {
                str(visible_file): "created",
                str(hidden_file): "created",
            }
        )

        mock_index_manager.index_document.assert_called_once_with(str(visible_file))

    @pytest.mark.asyncio
    async def test_batch_process_enqueues_index_when_task_mode_enabled(
        self, tmp_path, mock_index_manager
    ):
        docs_path = tmp_path / "docs"
        docs_path.mkdir()
        watcher = FileWatcher(
            documents_path=str(docs_path),
            index_manager=mock_index_manager,
            use_tasks=True,
        )

        with patch(
            "src.indexing.tasks.submit_index_request",
            return_value=TaskSubmissionResult(status="enqueued"),
        ) as enqueue:
            await watcher._batch_process({str(docs_path / "note.md"): "created"})

        enqueue.assert_called_once_with(str(docs_path / "note.md"))
        mock_index_manager.index_document.assert_not_called()

    @pytest.mark.asyncio
    async def test_batch_process_enqueues_remove_when_task_mode_enabled(
        self, tmp_path, mock_index_manager
    ):
        docs_path = tmp_path / "docs"
        docs_path.mkdir()
        watcher = FileWatcher(
            documents_path=str(docs_path),
            index_manager=mock_index_manager,
            use_tasks=True,
        )
        deleted_file = docs_path / "nested" / "note.md"
        deleted_file.parent.mkdir()

        with patch(
            "src.indexing.tasks.submit_remove_request",
            return_value=TaskSubmissionResult(status="enqueued"),
        ) as enqueue:
            await watcher._batch_process({str(deleted_file): "deleted"})

        enqueue.assert_called_once_with("nested/note")
        mock_index_manager.remove_document.assert_not_called()

    @pytest.mark.asyncio
    async def test_batch_process_requeues_when_task_queue_backpressured(
        self, tmp_path, mock_index_manager
    ):
        docs_path = tmp_path / "docs"
        docs_path.mkdir()
        watcher = FileWatcher(
            documents_path=str(docs_path),
            index_manager=mock_index_manager,
            use_tasks=True,
        )

        with patch(
            "src.indexing.tasks.submit_index_request",
            return_value=TaskSubmissionResult(status="backpressured"),
        ):
            await watcher._batch_process({str(docs_path / "note.md"): "created"})

        assert watcher._event_queue.get_nowait() == (
            "created",
            str(docs_path / "note.md"),
        )
        mock_index_manager.index_document.assert_not_called()

    @pytest.mark.asyncio
    async def test_batch_process_falls_back_to_direct_index_when_queue_unavailable(
        self, tmp_path, mock_index_manager
    ):
        docs_path = tmp_path / "docs"
        docs_path.mkdir()
        watcher = FileWatcher(
            documents_path=str(docs_path),
            index_manager=mock_index_manager,
            use_tasks=True,
        )

        with patch(
            "src.indexing.tasks.submit_index_request",
            return_value=TaskSubmissionResult(status="unavailable"),
        ):
            await watcher._batch_process({str(docs_path / "note.md"): "created"})

        mock_index_manager.index_document.assert_called_once_with(
            str(docs_path / "note.md")
        )


class TestFileWatcherDebounce:
    @pytest.mark.asyncio
    async def test_process_events_last_request_wins_per_file(
        self, tmp_path, mock_index_manager
    ):
        docs_path = tmp_path / "docs"
        docs_path.mkdir()
        watcher = FileWatcher(
            documents_path=str(docs_path),
            index_manager=mock_index_manager,
            cooldown=0.01,
        )

        observed: list[dict[str, str]] = []

        async def _record(events):
            observed.append(events.copy())

        watcher._running = True
        watcher._batch_process = _record  # type: ignore[method-assign]

        task = asyncio.create_task(watcher._process_events())
        watcher._event_queue.put_nowait(("created", str(docs_path / "note.md")))
        watcher._event_queue.put_nowait(("modified", str(docs_path / "note.md")))

        await asyncio.sleep(0.05)
        watcher._running = False
        await task

        assert observed == [{str(docs_path / "note.md"): "modified"}]

    def test_compute_doc_id_for_event_uses_matching_root(
        self, tmp_path, mock_index_manager
    ):
        project_a = tmp_path / "project_a"
        project_b = tmp_path / "project_b"
        project_a.mkdir()
        file_path = project_b / "docs" / "guide.md"
        file_path.parent.mkdir(parents=True)

        watcher = FileWatcher(
            documents_path=str(tmp_path),
            documents_paths=[str(project_a), str(project_b)],
            index_manager=mock_index_manager,
        )

        assert watcher._compute_doc_id_for_event(str(file_path)) == "project_b/docs/guide"

    def test_get_stats_reports_debounce_and_backpressure_counters(
        self, tmp_path, mock_index_manager
    ):
        docs_path = tmp_path / "docs"
        docs_path.mkdir()
        watcher = FileWatcher(
            documents_path=str(docs_path),
            documents_paths=[str(docs_path), str(tmp_path / "other")],
            index_manager=mock_index_manager,
        )
        watcher._events_received = 5
        watcher._events_processed = 3
        watcher._debounce_overwrites = 2
        watcher._deferred_task_retries = 1
        watcher._pending_debounce_count = 4
        watcher._watched_dirs = {"/tmp/a", "/tmp/b"}
        watcher._last_sync_time = "2026-03-17T00:00:00+00:00"

        stats = watcher.get_stats()

        assert stats.roots_count == 2
        assert stats.watched_dirs_count == 2
        assert stats.events_received == 5
        assert stats.events_processed == 3
        assert stats.debounce_overwrites == 2
        assert stats.deferred_task_retries == 1
        assert stats.pending_debounce_count == 4


class TestIsExcludedDir:
    """Tests for is_excluded_dir helper."""

    def test_hidden_dir_excluded_by_default(self):
        """Hidden directories should be excluded when exclude_hidden_dirs=True."""
        assert is_excluded_dir("/project/.git", [], exclude_hidden_dirs=True) is True
        assert is_excluded_dir("/project/.venv", [], exclude_hidden_dirs=True) is True

    def test_hidden_dir_allowed_when_disabled(self):
        """Hidden directories should be allowed when exclude_hidden_dirs=False."""
        assert is_excluded_dir("/project/.git", [], exclude_hidden_dirs=False) is False

    def test_exclude_pattern_matches(self):
        """Directories matching exclude patterns should be excluded."""
        patterns = ["**/node_modules/**", "**/.venv/**"]
        assert (
            is_excluded_dir(
                "/project/node_modules", patterns, exclude_hidden_dirs=False
            )
            is True
        )

    def test_normal_dir_not_excluded(self):
        """Normal directories should not be excluded."""
        patterns = ["**/node_modules/**", "**/.venv/**"]
        assert (
            is_excluded_dir("/project/docs", patterns, exclude_hidden_dirs=True)
            is False
        )
        assert (
            is_excluded_dir("/project/src/utils", patterns, exclude_hidden_dirs=True)
            is False
        )

    def test_nested_exclude_pattern(self):
        """Deeply nested excluded dirs should be detected."""
        patterns = ["**/build/**"]
        assert (
            is_excluded_dir(
                "/project/libs/mylib/build", patterns, exclude_hidden_dirs=False
            )
            is True
        )


class TestWalkIncludedDirs:
    """Tests for walk_included_dirs helper."""

    def test_returns_root(self, tmp_path):
        """Should always include the root directory."""
        result = walk_included_dirs(tmp_path, [], exclude_hidden_dirs=True)
        assert tmp_path in result

    def test_includes_normal_subdirs(self, tmp_path):
        """Should include non-excluded subdirectories."""
        (tmp_path / "docs").mkdir()
        (tmp_path / "src").mkdir()
        result = walk_included_dirs(tmp_path, [], exclude_hidden_dirs=True)
        assert tmp_path / "docs" in result
        assert tmp_path / "src" in result

    def test_excludes_hidden_dirs(self, tmp_path):
        """Should exclude hidden directories."""
        (tmp_path / ".git").mkdir()
        (tmp_path / ".venv").mkdir()
        (tmp_path / "docs").mkdir()
        result = walk_included_dirs(tmp_path, [], exclude_hidden_dirs=True)
        assert tmp_path / ".git" not in result
        assert tmp_path / ".venv" not in result
        assert tmp_path / "docs" in result

    def test_excludes_pattern_matched_dirs(self, tmp_path):
        """Should exclude directories matching exclude patterns."""
        (tmp_path / "node_modules").mkdir()
        (tmp_path / "node_modules" / "pkg").mkdir()
        (tmp_path / "src").mkdir()
        patterns = ["**/node_modules/**"]
        result = walk_included_dirs(tmp_path, patterns, exclude_hidden_dirs=False)
        assert tmp_path / "node_modules" not in result
        assert tmp_path / "node_modules" / "pkg" not in result
        assert tmp_path / "src" in result

    def test_prunes_subtrees(self, tmp_path):
        """Should prune entire excluded subtrees, not just the top dir."""
        venv = tmp_path / ".venv"
        venv.mkdir()
        (venv / "lib").mkdir()
        (venv / "lib" / "site-packages").mkdir()
        result = walk_included_dirs(tmp_path, [], exclude_hidden_dirs=True)
        assert all(".venv" not in str(p) for p in result)

    def test_includes_nested_dirs(self, tmp_path):
        """Should include deeply nested non-excluded directories."""
        (tmp_path / "a" / "b" / "c").mkdir(parents=True)
        result = walk_included_dirs(tmp_path, [], exclude_hidden_dirs=True)
        assert tmp_path / "a" in result
        assert tmp_path / "a" / "b" in result
        assert tmp_path / "a" / "b" / "c" in result
