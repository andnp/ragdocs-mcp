from __future__ import annotations

import asyncio
import logging
import queue
import threading
import time
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, cast

from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

from searchkernel.api import (
    PARSER_SUFFIXES,
    compute_doc_id_multi_root,
    should_include_file,
    walk_dirs_with_files,
)
from mcp_markdown_ragdocs.indexing.record_manager import RecordIndexManager as IndexManager
from mcp_markdown_ragdocs.coordination.task_submission import (
    TaskSubmissionPort,
)

logger = logging.getLogger(__name__)

type EventType = Literal["created", "modified", "deleted"]

# Maximum queue size to prevent memory exhaustion under load
MAX_QUEUE_SIZE = 1000

# Recursive root watches register one kernel inotify watch per directory
# under the root regardless of whether that directory has parseable files,
# so they never reduce total watch-descriptor consumption relative to
# watching only the directories that actually have parseable files. We
# always watch candidate directories individually; if that exhausts the
# kernel's inotify watch limit, scheduling fails per-directory in start()
# and refresh_watches(), logging a warning rather than crashing the watcher.


@dataclass(frozen=True)
class WatcherStats:
    roots_count: int
    watched_dirs_count: int
    queue_size: int
    pending_debounce_count: int
    events_received: int
    events_processed: int
    debounce_overwrites: int
    deferred_task_retries: int
    dropped_event_count: int
    dropped_since_reconcile: int
    last_sync_time: str | None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class FileWatcher:
    def __init__(
        self,
        documents_path: str,
        index_manager: IndexManager,
        cooldown: float = 0.5,
        documents_paths: list[str] | None = None,
        include_patterns: list[str] | None = None,
        exclude_patterns: list[str] | None = None,
        exclude_hidden_dirs: bool = True,
        parser_suffixes: set[str] | frozenset[str] | None = None,
        use_tasks: bool = False,
        task_backpressure_limit: int | None = None,
        on_overflow_detected: Callable[[], Awaitable[None]] | None = None,
        task_submission: TaskSubmissionPort | None = None,
    ):
        self._documents_path = Path(documents_path)
        self._documents_paths = (
            [Path(path) for path in documents_paths]
            if documents_paths
            else [self._documents_path]
        )
        self._index_manager = index_manager
        self._cooldown = cooldown
        self._include_patterns = include_patterns or ["**/*"]
        self._exclude_patterns = exclude_patterns or []
        self._exclude_hidden_dirs = exclude_hidden_dirs
        self._parser_suffixes: set[str] = set(parser_suffixes) if parser_suffixes else set(PARSER_SUFFIXES)
        self._observer: Observer | None = None  # type: ignore[reportInvalidTypeForm]
        # Bounded queue prevents memory exhaustion during high file change rates
        self._event_queue: queue.Queue = queue.Queue(maxsize=MAX_QUEUE_SIZE)  # items: tuple[EventType, str]
        self._running = False
        self._task: asyncio.Task[None] | None = None
        self._last_sync_time: str | None = None
        self._stopped_cleanly: bool = True
        self._event_handler: _DocumentEventHandler | None = None
        self._watched_dirs: set[str] = set()
        self._use_tasks = use_tasks
        self._task_backpressure_limit = task_backpressure_limit
        self._task_submission = task_submission
        self._events_received = 0
        self._events_processed = 0
        self._debounce_overwrites = 0
        self._deferred_task_retries = 0
        self._pending_debounce_count = 0
        self._on_overflow_detected = on_overflow_detected

    @property
    def stopped_cleanly(self) -> bool:
        """Return whether the watcher was stopped cleanly without timeout."""
        return self._stopped_cleanly

    def __del__(self) -> None:
        """Safety net to ensure observer is cleaned up on garbage collection."""
        try:
            if self._observer is not None:
                self._observer.unschedule_all()
                self._observer.stop()
                self._observer = None
        except Exception:  # noqa: S110, BLE001 -- __del__ runs during GC/interpreter
            # shutdown, when the logging module itself may already be torn
            # down; swallowing silently here is safer than risking a
            # secondary exception from logging.
            pass

    def start(self):
        if self._running:
            return

        watched_dirs: list[Path] = []
        seen_dirs: set[Path] = set()
        for root in self._documents_paths:
            if not root.exists():
                continue
            for dir_path in walk_dirs_with_files(
                root,
                self._exclude_patterns,
                self._exclude_hidden_dirs,
                self._parser_suffixes,
            ):
                if dir_path not in seen_dirs:
                    seen_dirs.add(dir_path)
                    watched_dirs.append(dir_path)

        self._running = True
        self._event_handler = _DocumentEventHandler(
            self._event_queue,
            self._parser_suffixes,
            exclude_patterns=self._exclude_patterns,
            exclude_hidden_dirs=self._exclude_hidden_dirs,
        )
        observer = Observer()
        self._watched_dirs = set()
        failed_dirs = 0
        for path in watched_dirs:
            try:
                observer.schedule(self._event_handler, str(path), recursive=False)
                self._watched_dirs.add(str(path))
            except OSError as e:
                failed_dirs += 1
                logger.warning("Failed to schedule watch on %s: %s", path, e)
        observer.start()
        self._observer = observer
        self._task = asyncio.create_task(self._process_events())
        logger.info(
            "File watcher started for %d roots (%d directories with parseable files, %d watched)",
            len(self._documents_paths),
            len(watched_dirs),
            len(self._watched_dirs),
        )
        if failed_dirs:
            logger.warning(
                "File watcher failed to schedule %d of %d directories; the "
                "inotify watch limit may be too low "
                "(see /proc/sys/fs/inotify/max_user_watches)",
                failed_dirs,
                len(watched_dirs),
            )

    def refresh_watches(self) -> None:
        """Register inotify watches for any new directories that have appeared
        since startup or the last refresh.  Called after each reconciliation
        cycle so that newly-created directories are picked up without requiring
        a restart."""
        if not self._running or self._observer is None or self._event_handler is None:
            return

        current_dirs: list[Path] = []
        seen_dirs: set[Path] = set()
        for root in self._documents_paths:
            if not root.exists():
                continue
            for dir_path in walk_dirs_with_files(
                root,
                self._exclude_patterns,
                self._exclude_hidden_dirs,
                self._parser_suffixes,
            ):
                if dir_path not in seen_dirs:
                    seen_dirs.add(dir_path)
                    current_dirs.append(dir_path)
        new_dirs = [d for d in current_dirs if str(d) not in self._watched_dirs]
        for dir_path in new_dirs:
            try:
                self._observer.schedule(
                    self._event_handler, str(dir_path), recursive=False
                )
                self._watched_dirs.add(str(dir_path))
            except OSError as e:
                logger.warning("Failed to schedule watch on %s: %s", dir_path, e)

        if new_dirs:
            logger.info(
                "File watcher: added %d new watched directories (total: %d)",
                len(new_dirs),
                len(self._watched_dirs),
            )

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
            # First unschedule all watches to stop receiving events
            try:
                self._observer.unschedule_all()
            except Exception as e:
                logger.warning("Failed to unschedule watches: %s", e, exc_info=True)

            # Then stop the observer thread
            self._observer.stop()

            # Join with timeout - don't wait forever
            try:
                await asyncio.wait_for(
                    asyncio.to_thread(self._observer.join, timeout=1.0),
                    timeout=1.5,
                )
            except TimeoutError:
                self._stopped_cleanly = False
                logger.warning(
                    "Observer thread did not stop within timeout, forcing unschedule"
                )
                # Try unschedule again in case it helps
                try:
                    self._observer.unschedule_all()
                except Exception:
                    logger.debug(
                        "Forced unschedule_all also failed", exc_info=True
                    )

            # Mark for garbage collection
            self._observer = None

        # Drain remaining events from queue (with timeout)
        try:
            await asyncio.wait_for(self._drain_queue(), timeout=2.0)
        except TimeoutError:
            remaining = self._event_queue.qsize()
            if remaining > 0:
                logger.warning(f"Queue drain timed out, {remaining} events lost")

        # Cancel the event processing task
        if self._task:
            self._task.cancel()
            try:
                await asyncio.wait_for(self._task, timeout=1.0)
            except (TimeoutError, asyncio.CancelledError):
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
        pending_events: dict[str, tuple[EventType, float]] = {}

        while self._running:
            try:
                try:
                    # Use timeout on queue.get to allow checking _running flag
                    event_type, file_path = await asyncio.to_thread(
                        self._event_queue.get, timeout=0.1
                    )
                    if file_path:
                        self._events_received += 1
                        if file_path in pending_events:
                            self._debounce_overwrites += 1
                        pending_events[file_path] = (
                            cast(EventType, event_type),
                            time.monotonic(),
                        )
                        self._pending_debounce_count = len(pending_events)
                except queue.Empty:
                    pass

                if pending_events:
                    now = time.monotonic()
                    ready_events: dict[str, EventType] = {
                        file_path: event_type
                        for file_path, (event_type, last_seen)
                        in pending_events.items()
                        if (now - last_seen) >= self._cooldown
                    }
                    if ready_events:
                        await self._batch_process(ready_events)
                        for file_path in ready_events:
                            pending_events.pop(file_path, None)
                        self._pending_debounce_count = len(pending_events)
            except asyncio.CancelledError:
                break
            except Exception as e:  # noqa: BLE001 -- background loop must not crash the watcher
                logger.error(f"Error in event processing: {e}")

        # Process remaining events with timeout
        if pending_events:
            try:
                await asyncio.wait_for(
                    self._batch_process(
                        {
                            file_path: event_type
                            for file_path, (event_type, _)
                            in pending_events.items()
                        }
                    ),
                    timeout=1.0,
                )
                self._pending_debounce_count = 0
            except Exception as e:  # noqa: BLE001 -- final shutdown drain must not crash the watcher
                logger.warning(f"Failed to process final events: {e}")

    async def _batch_process(self, events: dict[str, EventType]):
        deferred_events: dict[str, EventType] = {}
        self._events_processed += len(events)
        task_index_paths: list[str] = []
        task_remove_doc_ids: list[str] = []
        deleted_paths_by_doc_id: dict[str, str] = {}
        direct_index_paths: list[str] = []
        direct_remove_doc_ids: list[str] = []

        for file_path, event_type in events.items():
            # Filter out excluded files before processing
            if not should_include_file(
                file_path,
                self._include_patterns,
                self._exclude_patterns,
                self._exclude_hidden_dirs,
                documents_roots=[str(path) for path in self._documents_paths],
            ):
                logger.debug(f"Skipping excluded file: {file_path}")
                continue

            try:
                if event_type in ("created", "modified"):
                    if self._use_tasks:
                        task_index_paths.append(file_path)
                        continue
                    direct_index_paths.append(file_path)
                elif event_type == "deleted":
                    doc_id = self._compute_doc_id_for_event(file_path)
                    if self._use_tasks:
                        task_remove_doc_ids.append(doc_id)
                        deleted_paths_by_doc_id.setdefault(doc_id, file_path)
                        continue
                    direct_remove_doc_ids.append(doc_id)
            except Exception as e:  # noqa: BLE001 -- per-file processing must not crash the batch
                logger.error(f"Failed to process {file_path}: {e}")

        if self._use_tasks:
            if self._task_submission is None:
                from mcp_markdown_ragdocs.indexing.tasks import (
                    submit_index_request_batch,
                    submit_remove_request_batch,
                )

                submit_index = submit_index_request_batch
                submit_remove = submit_remove_request_batch
            else:
                submit_index = self._task_submission.submit_index_request_batch
                submit_remove = self._task_submission.submit_remove_request_batch

            if task_index_paths:
                submission = submit_index(task_index_paths)
                if not submission.queue_available:
                    direct_index_paths.extend(list(dict.fromkeys(task_index_paths)))
                else:
                    if submission.enqueued_count > 0:
                        logger.info(
                            "Enqueued watcher indexing batch for %d file(s)",
                            submission.enqueued_count,
                        )
                    for file_path in submission.backpressured_items:
                        deferred_events[file_path] = events[file_path]

            if task_remove_doc_ids:
                submission = submit_remove(task_remove_doc_ids)
                if not submission.queue_available:
                    direct_remove_doc_ids.extend(list(dict.fromkeys(task_remove_doc_ids)))
                else:
                    if submission.enqueued_count > 0:
                        logger.info(
                            "Enqueued watcher removal batch for %d document(s)",
                            submission.enqueued_count,
                        )
                    for doc_id in submission.backpressured_items:
                        file_path = deleted_paths_by_doc_id.get(doc_id)
                        if file_path is not None:
                            deferred_events[file_path] = "deleted"

        await self._process_direct_batch(
            index_paths=direct_index_paths,
            remove_doc_ids=direct_remove_doc_ids,
        )

        self._last_sync_time = datetime.now(UTC).isoformat()

        for file_path, event_type in deferred_events.items():
            try:
                self._deferred_task_retries += 1
                self._event_queue.put_nowait((event_type, file_path))
            except queue.Full:
                logger.warning(
                    "Watcher queue full while deferring %s for backpressure retry",
                    file_path,
                )

        await self._maybe_trigger_overflow_reconciliation()

    async def _maybe_trigger_overflow_reconciliation(self) -> None:
        """Check for inotify queue overflow and trigger reconciliation if detected.

        If events were dropped since the last reconciliation and a callback is
        registered, invoke the callback to trigger a full reconciliation pass.
        After callback returns, reset the per-reconciliation counter.
        """
        if self._on_overflow_detected is None:
            return

        if not self.should_reconcile():
            return

        try:
            await self._on_overflow_detected()
            self.reset_dropped_counter()
            logger.info(
                "Triggered reconciliation due to inotify queue overflow "
                "(%d events dropped since last reconciliation)",
                self.dropped_since_reconcile,
            )
        except Exception:
            logger.exception("Error invoking overflow reconciliation callback")

    async def _process_direct_batch(
        self,
        *,
        index_paths: list[str],
        remove_doc_ids: list[str],
    ) -> None:
        mutated = False

        if index_paths:
            unique_index_paths = list(dict.fromkeys(index_paths))
            try:
                await asyncio.to_thread(
                    self._index_manager.index_documents,
                    unique_index_paths,
                    force=False,
                    persist=False,
                )
                mutated = True
                logger.info("Indexed %d file(s) directly from watcher burst", len(unique_index_paths))
            except Exception:
                logger.warning(
                    "Direct watcher index batch failed; retrying files individually before one final persist",
                    exc_info=True,
                )
                for file_path in unique_index_paths:
                    try:
                        await asyncio.to_thread(
                            self._index_manager.index_document,
                            file_path,
                        )
                        mutated = True
                        logger.info(f"Indexed: {file_path}")
                    except Exception as e:  # noqa: BLE001 -- per-file processing must not crash the batch
                        logger.error(f"Failed to process {file_path}: {e}")

        if remove_doc_ids:
            unique_remove_doc_ids = list(dict.fromkeys(remove_doc_ids))
            try:
                await asyncio.to_thread(
                    self._index_manager.remove_documents,
                    unique_remove_doc_ids,
                    persist=False,
                )
                mutated = True
                logger.info(
                    "Removed %d document(s) directly from watcher burst",
                    len(unique_remove_doc_ids),
                )
            except Exception:
                logger.warning(
                    "Direct watcher remove batch failed; retrying removals individually before one final persist",
                    exc_info=True,
                )
                for doc_id in unique_remove_doc_ids:
                    try:
                        await asyncio.to_thread(
                            self._index_manager.remove_document,
                            doc_id,
                        )
                        mutated = True
                    except Exception as e:  # noqa: BLE001 -- per-doc removal must not crash the batch
                        logger.error(f"Failed to remove {doc_id}: {e}")

        if mutated:
            try:
                await asyncio.to_thread(self._index_manager.persist)
            except Exception as e:  # noqa: BLE001 -- burst persist must not crash the watcher
                logger.error(f"Failed to persist watcher burst changes: {e}")

    def get_stats(self) -> WatcherStats:
        return WatcherStats(
            roots_count=len(self._documents_paths),
            watched_dirs_count=len(self._watched_dirs),
            queue_size=self.get_pending_queue_size(),
            pending_debounce_count=self._pending_debounce_count,
            events_received=self._events_received,
            events_processed=self._events_processed,
            debounce_overwrites=self._debounce_overwrites,
            deferred_task_retries=self._deferred_task_retries,
            dropped_event_count=self.dropped_event_count,
            dropped_since_reconcile=self.dropped_since_reconcile,
            last_sync_time=self._last_sync_time,
        )

    def _compute_doc_id_for_event(self, file_path: str) -> str:
        try:
            return compute_doc_id_multi_root(Path(file_path).resolve(), self._documents_paths)
        except Exception:  # noqa: BLE001 -- fall back to stem for any path-resolution failure
            logger.warning(
                "Failed to compute doc_id for %s; falling back to file stem",
                file_path,
                exc_info=True,
            )
            return Path(file_path).stem

    def get_pending_queue_size(self) -> int:
        return self._event_queue.qsize()

    def get_last_sync_time(self) -> str | None:
        return self._last_sync_time

    def get_failed_files(self) -> list[dict[str, str]]:
        return self._index_manager.get_failed_files()

    @property
    def dropped_event_count(self) -> int:
        """Total number of events dropped due to queue full."""
        if self._event_handler is None:
            return 0
        return self._event_handler.dropped_event_count

    @property
    def dropped_since_reconcile(self) -> int:
        """Number of events dropped since last reconciliation."""
        if self._event_handler is None:
            return 0
        return self._event_handler.dropped_since_reconcile

    def reset_dropped_counter(self) -> None:
        """Call after reconciliation to reset per-reconcile counter."""
        if self._event_handler is not None:
            self._event_handler.reset_dropped_counter()

    def set_overflow_callback(
        self, callback: Callable[[], Awaitable[None]] | None
    ) -> None:
        """Set the callback to invoke when inotify queue overflow is detected.

        The callback should be an async function that triggers reconciliation.
        Pass None to disable overflow detection.
        """
        self._on_overflow_detected = callback

    def should_reconcile(self) -> bool:
        """True if drops occurred since last reconcile and reconciliation is advised."""
        return self.dropped_since_reconcile > 0


class _DocumentEventHandler(FileSystemEventHandler):
    def __init__(
        self,
        event_queue: queue.Queue,  # items: tuple[EventType, str]
        suffixes: set[str],
        exclude_patterns: list[str] | None = None,
        exclude_hidden_dirs: bool = True,
    ):
        super().__init__()
        self._queue = event_queue
        self._suffixes = suffixes
        self._exclude_patterns = exclude_patterns or []
        self._exclude_hidden_dirs = exclude_hidden_dirs
        self._lock = threading.Lock()
        self._dropped_events = 0  # Track backpressure events (total)
        self._dropped_since_last_reconcile = 0  # Track drops since last reconcile

    def _is_supported_file(self, path: str | bytes) -> bool:
        path_str = path if isinstance(path, str) else path.decode("utf-8")
        return Path(path_str).suffix.lower() in self._suffixes

    @property
    def dropped_event_count(self) -> int:
        """Total number of events dropped due to queue full."""
        with self._lock:
            return self._dropped_events

    @property
    def dropped_since_reconcile(self) -> int:
        """Number of events dropped since last reconciliation."""
        with self._lock:
            return self._dropped_since_last_reconcile

    def reset_dropped_counter(self) -> None:
        """Call after reconciliation to reset per-reconcile counter."""
        with self._lock:
            self._dropped_since_last_reconcile = 0

    def _queue_event(self, event_type: EventType, path_str: str):
        """Queue event with backpressure handling."""
        try:
            self._queue.put_nowait((event_type, path_str))
        except queue.Full:
            # Drop event if queue is full to prevent memory exhaustion
            with self._lock:
                self._dropped_events += 1
                self._dropped_since_last_reconcile += 1
                dropped_total = self._dropped_events
                dropped_since_reconcile = self._dropped_since_last_reconcile
            if dropped_total % 100 == 0:
                logger.warning(
                    "Event queue full (%d), dropped %d total events (%d since last reconcile). "
                    "Increase cooldown or MAX_QUEUE_SIZE if this persists.",
                    MAX_QUEUE_SIZE,
                    dropped_total,
                    dropped_since_reconcile,
                )

    def on_created(self, event: FileSystemEvent):
        # Directory creation is handled by refresh_watches() after reconciliation.
        if event.is_directory:
            return

        if self._is_supported_file(event.src_path):
            path_str = (
                event.src_path
                if isinstance(event.src_path, str)
                else event.src_path.decode("utf-8")
            )
            self._queue_event("created", path_str)

    def on_modified(self, event: FileSystemEvent):
        if not event.is_directory and self._is_supported_file(event.src_path):
            path_str = (
                event.src_path
                if isinstance(event.src_path, str)
                else event.src_path.decode("utf-8")
            )
            self._queue_event("modified", path_str)

    def on_deleted(self, event: FileSystemEvent):
        if not event.is_directory and self._is_supported_file(event.src_path):
            path_str = (
                event.src_path
                if isinstance(event.src_path, str)
                else event.src_path.decode("utf-8")
            )
            self._queue_event("deleted", path_str)
