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

if TYPE_CHECKING:
    from watchdog.observers.api import BaseObserver

logger = logging.getLogger(__name__)

EventType: TypeAlias = Literal["created", "modified", "deleted"]


class FileWatcher:
    def __init__(
        self, documents_path: str, index_manager: IndexManager, cooldown: float = 0.5
    ):
        self._documents_path = Path(documents_path)
        self._index_manager = index_manager
        self._cooldown = cooldown
        self._observer: BaseObserver | None = None
        self._event_queue = queue.Queue[tuple[EventType, str]]()
        self._running = False
        self._task: asyncio.Task[None] | None = None
        self._last_sync_time: str | None = None

    def start(self):
        if self._running:
            return

        self._running = True
        event_handler = _MarkdownEventHandler(self._event_queue)
        observer = Observer()
        observer.schedule(event_handler, str(self._documents_path), recursive=True)
        observer.start()
        self._observer = observer
        self._task = asyncio.create_task(self._process_events())
        logger.info(f"File watcher started for {self._documents_path}")

    async def stop(self):
        if not self._running:
            return

        self._running = False

        if self._observer:
            self._observer.stop()
            self._observer.join()
            self._observer = None

        if self._task:
            self._event_queue.put(("deleted", ""))
            try:
                await asyncio.wait_for(self._task, timeout=5.0)
            except asyncio.TimeoutError:
                self._task.cancel()
                try:
                    await self._task
                except asyncio.CancelledError:
                    pass
            self._task = None

        logger.info("File watcher stopped")

    async def _process_events(self):
        events: dict[str, EventType] = {}

        while self._running:
            try:
                event_type, file_path = await asyncio.to_thread(
                    self._event_queue.get, timeout=self._cooldown
                )

                if not file_path:
                    continue

                events[file_path] = event_type
            except queue.Empty:
                if events:
                    await self._batch_process(events)
                    events = {}

        if events:
            await self._batch_process(events)

    async def _batch_process(self, events: dict[str, EventType]):
        for file_path, event_type in events.items():
            try:
                if event_type in ("created", "modified"):
                    await asyncio.to_thread(
                        self._index_manager.index_document, file_path
                    )
                    logger.info(f"Indexed: {file_path}")
                elif event_type == "deleted":
                    doc_id = Path(file_path).stem
                    await asyncio.to_thread(
                        self._index_manager.remove_document, doc_id
                    )
                    logger.info(f"Removed: {file_path}")
            except Exception as e:
                logger.error(f"Failed to process {file_path}: {e}")

        self._last_sync_time = datetime.now(timezone.utc).isoformat()

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

    def _is_markdown(self, path: str | bytes):
        path_str = path if isinstance(path, str) else path.decode("utf-8")
        return Path(path_str).suffix.lower() in (".md", ".markdown")

    def on_created(self, event: FileSystemEvent):
        if not event.is_directory and self._is_markdown(event.src_path):
            path_str = (
                event.src_path
                if isinstance(event.src_path, str)
                else event.src_path.decode("utf-8")
            )
            self._queue.put_nowait(("created", path_str))

    def on_modified(self, event: FileSystemEvent):
        if not event.is_directory and self._is_markdown(event.src_path):
            path_str = (
                event.src_path
                if isinstance(event.src_path, str)
                else event.src_path.decode("utf-8")
            )
            self._queue.put_nowait(("modified", path_str))

    def on_deleted(self, event: FileSystemEvent):
        if not event.is_directory and self._is_markdown(event.src_path):
            path_str = (
                event.src_path
                if isinstance(event.src_path, str)
                else event.src_path.decode("utf-8")
            )
            self._queue.put_nowait(("deleted", path_str))
