from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import logging
import queue
from pathlib import Path
from typing import TYPE_CHECKING, Literal, TypeAlias

from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

from src.indexing.manager import IndexManager
from src.utils import should_include_file

if TYPE_CHECKING:
    from watchdog.observers.api import BaseObserver

logger = logging.getLogger(__name__)

EventType: TypeAlias = Literal["created", "modified", "deleted"]

# Maximum queue size to prevent memory exhaustion under load
MAX_QUEUE_SIZE = 1000


def _count_directories(path: Path) -> int:
    """Count directories under path for inotify limit estimation."""
    try:
        count = 1  # Include the root
        for item in path.rglob("*"):
            if item.is_dir():
                count += 1
        return count
    except OSError:
        return -1  # Can't count, likely permission issue


class FileWatcher:
    def __init__(
        self,
        documents_path: str,
        index_manager: IndexManager,
        cooldown: float = 0.5,
        include_patterns: list[str] | None = None,
        exclude_patterns: list[str] | None = None,
        exclude_hidden_dirs: bool = True,
    ):
        self._documents_path = Path(documents_path)
        self._index_manager = index_manager
        self._cooldown = cooldown
        self._include_patterns = include_patterns or ["**/*"]
        self._exclude_patterns = exclude_patterns or []
        self._exclude_hidden_dirs = exclude_hidden_dirs
        self._observer: BaseObserver | None = None
        # Bounded queue prevents memory exhaustion during high file change rates
        self._event_queue = queue.Queue[tuple[EventType, str]](maxsize=MAX_QUEUE_SIZE)
        self._running = False
        self._task: asyncio.Task[None] | None = None
        self._last_sync_time: str | None = None

    def start(self):
        if self._running:
            return

        dir_count = _count_directories(self._documents_path)
        if dir_count > 1000:
            logger.warning(
                "Large directory tree detected (%d dirs). "
                "May exhaust inotify watches. Consider setting "
                "[tool.ragdocs.worker] enabled = true in pyproject.toml "
                "to use multiprocess mode with reduced watching.",
                dir_count
            )

        self._running = True
        event_handler = _MarkdownEventHandler(self._event_queue)
        observer = Observer()
        observer.schedule(event_handler, str(self._documents_path), recursive=True)
        observer.start()
        self._observer = observer
        self._task = asyncio.create_task(self._process_events())
        logger.info(f"File watcher started for {self._documents_path}")

    async def stop(self):
        """Stop file watcher and drain remaining events before shutdown.

        Ensures queued file system events are processed before termination
        to prevent data loss during shutdown.
        """
        if not self._running:
            return

        # Set flag to prevent new event acceptance
        self._running = False

        # Stop the observer thread (no new events)
        if self._observer:
            self._observer.stop()
            # Join with short timeout - don't wait forever
            try:
                await asyncio.wait_for(
                    asyncio.to_thread(self._observer.join, timeout=1.0),
                    timeout=1.5,
                )
            except asyncio.TimeoutError:
                logger.warning("Observer thread did not stop within timeout")
            self._observer = None

        # Drain remaining events from queue (with timeout)
        try:
            await asyncio.wait_for(self._drain_queue(), timeout=2.0)
        except asyncio.TimeoutError:
            remaining = self._event_queue.qsize()
            if remaining > 0:
                logger.warning(f"Queue drain timed out, {remaining} events lost")

        # Cancel the event processing task
        if self._task:
            self._task.cancel()
            try:
                await asyncio.wait_for(self._task, timeout=1.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                pass
            self._task = None

        logger.info("File watcher stopped")

    async def _drain_queue(self):
        """Process all remaining events in queue before shutdown."""
        events: dict[str, EventType] = {}

        # Collect all queued events (non-blocking)
        while True:
            try:
                event_type, file_path = await asyncio.to_thread(
                    self._event_queue.get, timeout=0.1
                )
                if file_path:
                    events[file_path] = event_type
            except queue.Empty:
                break

        # Process collected events if any
        if events:
            logger.info(f"Draining {len(events)} queued file system events")
            await self._batch_process(events)

    async def _process_events(self):
        events: dict[str, EventType] = {}

        while self._running:
            try:
                try:
                    # Use timeout on queue.get to allow checking _running flag
                    event_type, file_path = await asyncio.to_thread(
                        self._event_queue.get, timeout=0.5
                    )
                    if file_path:
                        events[file_path] = event_type
                except queue.Empty:
                    if events:
                        await self._batch_process(events)
                        events = {}
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in event processing: {e}")

        # Process remaining events with timeout
        if events:
            try:
                await asyncio.wait_for(self._batch_process(events), timeout=1.0)
            except (asyncio.TimeoutError, Exception) as e:
                logger.warning(f"Failed to process final events: {e}")

    async def _batch_process(self, events: dict[str, EventType]):
        for file_path, event_type in events.items():
            # Filter out excluded files before processing
            if not should_include_file(
                file_path,
                self._include_patterns,
                self._exclude_patterns,
                self._exclude_hidden_dirs,
            ):
                logger.debug(f"Skipping excluded file: {file_path}")
                continue

            try:
                if event_type in ("created", "modified"):
                    await asyncio.to_thread(
                        self._index_manager.index_document, file_path
                    )
                    logger.info(f"Indexed: {file_path}")
                elif event_type == "deleted":
                    # Compute doc_id same way as IndexManager
                    try:
                        rel_path = Path(file_path).relative_to(self._documents_path)
                        doc_id = str(rel_path.with_suffix(""))
                    except ValueError:
                        doc_id = Path(file_path).stem
                    await asyncio.to_thread(
                        self._index_manager.remove_document, doc_id
                    )
                    logger.info(f"Removed: {file_path}")
            except Exception as e:
                logger.error(f"Failed to process {file_path}: {e}")

        self._last_sync_time = datetime.now(timezone.utc).isoformat()

    def get_pending_queue_size(self) -> int:
        return self._event_queue.qsize()

    def get_last_sync_time(self) -> str | None:
        return self._last_sync_time

    def get_failed_files(self) -> list[dict[str, str]]:
        return self._index_manager.get_failed_files()


class _MarkdownEventHandler(FileSystemEventHandler):
    def __init__(self, queue: queue.Queue[tuple[EventType, str]]):
        super().__init__()
        self._queue = queue
        self._dropped_events = 0  # Track backpressure events

    def _is_markdown(self, path: str | bytes):
        path_str = path if isinstance(path, str) else path.decode("utf-8")
        return Path(path_str).suffix.lower() in (".md", ".markdown")

    def _queue_event(self, event_type: EventType, path_str: str):
        """Queue event with backpressure handling."""
        try:
            self._queue.put_nowait((event_type, path_str))
        except queue.Full:
            # Drop event if queue is full to prevent memory exhaustion
            self._dropped_events += 1
            if self._dropped_events % 100 == 1:  # Log every 100th drop to avoid spam
                logger.warning(
                    f"Event queue full ({MAX_QUEUE_SIZE}), dropped {self._dropped_events} events. "
                    "Increase cooldown or MAX_QUEUE_SIZE if this persists."
                )

    def on_created(self, event: FileSystemEvent):
        if not event.is_directory and self._is_markdown(event.src_path):
            path_str = (
                event.src_path
                if isinstance(event.src_path, str)
                else event.src_path.decode("utf-8")
            )
            self._queue_event("created", path_str)

    def on_modified(self, event: FileSystemEvent):
        if not event.is_directory and self._is_markdown(event.src_path):
            path_str = (
                event.src_path
                if isinstance(event.src_path, str)
                else event.src_path.decode("utf-8")
            )
            self._queue_event("modified", path_str)

    def on_deleted(self, event: FileSystemEvent):
        if not event.is_directory and self._is_markdown(event.src_path):
            path_str = (
                event.src_path
                if isinstance(event.src_path, str)
                else event.src_path.decode("utf-8")
            )
            self._queue_event("deleted", path_str)
