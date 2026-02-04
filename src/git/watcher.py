"""Git directory file watcher for automatic commit indexing."""

from __future__ import annotations

import asyncio
import logging
import queue
from pathlib import Path
from typing import TYPE_CHECKING

from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

from src.config import Config
from src.git.commit_indexer import CommitIndexer

if TYPE_CHECKING:
    from watchdog.observers.api import BaseObserver

logger = logging.getLogger(__name__)

# Maximum queue size to prevent memory exhaustion under load
MAX_QUEUE_SIZE = 1000


class GitWatcher:
    """Watches .git directories for changes and triggers incremental indexing."""

    def __init__(
        self,
        git_repos: list[Path],
        commit_indexer: CommitIndexer,
        config: Config,
        cooldown: float = 5.0,
    ):
        """
        Initialize git watcher.

        Args:
            git_repos: List of .git directory paths to watch
            commit_indexer: CommitIndexer instance
            config: Configuration object
            cooldown: Debounce cooldown in seconds
        """
        self._git_repos = git_repos
        self._commit_indexer = commit_indexer
        self._config = config
        self._cooldown = cooldown
        self._observers: list[BaseObserver] = []
        self._event_queue: queue.Queue[Path] = queue.Queue(maxsize=MAX_QUEUE_SIZE)
        self._running = False
        self._task: asyncio.Task[None] | None = None
        self._last_indexed_timestamp: float | None = None
        
        # Dropped event tracking
        self._dropped_events: int = 0
        self._dropped_log_threshold: int = 10  # Log every N drops

    @property
    def dropped_event_count(self) -> int:
        """Number of events dropped due to queue backpressure."""
        return self._dropped_events

    def reset_dropped_counter(self) -> None:
        """Reset the dropped event counter."""
        self._dropped_events = 0

    def should_catchup(self) -> bool:
        """True if drops occurred and catch-up indexing is advised."""
        return self._dropped_events > 0

    async def run_catchup(self) -> int:
        """Re-index recent commits that may have been missed due to queue drops.

        Returns number of commits indexed.
        """
        from src.git.parallel_indexer import (
            ParallelIndexingConfig,
            index_commits_parallel,
        )
        from src.git.repository import get_commits_after_timestamp

        total_indexed = 0
        parallel_config = ParallelIndexingConfig(
            max_workers=self._config.git_indexing.parallel_workers,
            batch_size=self._config.git_indexing.batch_size,
            embed_batch_size=self._config.git_indexing.embed_batch_size,
        )

        for git_dir in self._git_repos:
            try:
                last_timestamp = await asyncio.to_thread(
                    self._commit_indexer.get_last_indexed_timestamp,
                    str(git_dir),
                )

                commit_hashes = await asyncio.to_thread(
                    get_commits_after_timestamp,
                    git_dir,
                    last_timestamp,
                )

                if not commit_hashes:
                    continue

                logger.info(
                    "Catch-up indexing %d commits from %s",
                    len(commit_hashes),
                    git_dir.parent,
                )

                indexed = await index_commits_parallel(
                    commit_hashes,
                    git_dir,
                    self._commit_indexer,
                    parallel_config,
                    self._config.git_indexing.delta_max_lines,
                )
                total_indexed += indexed

            except Exception as e:
                logger.error(
                    "Catch-up indexing failed for %s: %s", git_dir, e, exc_info=True
                )

        self.reset_dropped_counter()
        return total_indexed

    def start(self) -> None:
        """Start watching git directories."""
        if self._running:
            return

        self._running = True

        for git_dir in self._git_repos:
            # Watch specific paths: HEAD and refs/
            watch_paths = [
                git_dir / "HEAD",
                git_dir / "refs",
            ]

            for watch_path in watch_paths:
                if watch_path.exists():
                    event_handler = _GitEventHandler(
                        watcher=self,
                        event_queue=self._event_queue,
                        git_dir=git_dir,
                    )
                    observer = Observer()
                    observer.schedule(
                        event_handler,
                        str(watch_path),
                        recursive=(watch_path.name == "refs"),
                    )
                    observer.start()
                    self._observers.append(observer)

        self._task = asyncio.create_task(self._process_events())
        logger.info(f"Git watcher started for {len(self._git_repos)} repositories")

    async def stop(self) -> None:
        """Stop watching git directories."""
        if not self._running:
            return

        self._running = False

        # Stop all observers
        for observer in self._observers:
            observer.stop()
            try:
                await asyncio.wait_for(
                    asyncio.to_thread(observer.join, timeout=1.0),
                    timeout=1.5,
                )
            except TimeoutError:
                logger.warning("Observer thread did not stop within timeout")

        self._observers.clear()

        # Cancel processing task
        if self._task:
            self._task.cancel()
            try:
                await asyncio.wait_for(self._task, timeout=1.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                pass
            self._task = None

        logger.info("Git watcher stopped")

    async def _process_events(self) -> None:
        """Process queued git directory changes with debouncing."""
        pending_repos: set[Path] = set()

        while self._running:
            try:
                try:
                    git_dir = await asyncio.to_thread(
                        self._event_queue.get, timeout=0.5
                    )
                    pending_repos.add(git_dir)
                except queue.Empty:
                    if pending_repos:
                        await asyncio.sleep(self._cooldown)
                        await self._batch_process(pending_repos)
                        pending_repos.clear()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in git event processing: {e}")

        # Process remaining
        if pending_repos:
            try:
                await asyncio.wait_for(
                    self._batch_process(pending_repos),
                    timeout=5.0,
                )
            except (asyncio.TimeoutError, Exception) as e:
                logger.warning(f"Failed to process final git events: {e}")

    async def _batch_process(self, git_dirs: set[Path]) -> None:
        """Incrementally index commits for changed repositories."""
        from src.git.parallel_indexer import (
            ParallelIndexingConfig,
            index_commits_parallel,
        )
        from src.git.repository import get_commits_after_timestamp

        parallel_config = ParallelIndexingConfig(
            max_workers=self._config.git_indexing.parallel_workers,
            batch_size=self._config.git_indexing.batch_size,
            embed_batch_size=self._config.git_indexing.embed_batch_size,
        )

        for git_dir in git_dirs:
            try:
                last_timestamp = await asyncio.to_thread(
                    self._commit_indexer.get_last_indexed_timestamp,
                    str(git_dir),
                )

                commit_hashes = await asyncio.to_thread(
                    get_commits_after_timestamp,
                    git_dir,
                    last_timestamp,
                )

                if not commit_hashes:
                    logger.debug(f"No new commits in {git_dir.parent}")
                    continue

                logger.info(f"Indexing {len(commit_hashes)} new commits from {git_dir.parent}")

                indexed = await index_commits_parallel(
                    commit_hashes,
                    git_dir,
                    self._commit_indexer,
                    parallel_config,
                    self._config.git_indexing.delta_max_lines,
                )

                logger.info(f"Updated commit index for {git_dir.parent.name}: {indexed} commits")

            except Exception as e:
                logger.error(f"Failed to update commits for {git_dir}: {e}", exc_info=True)


class _GitEventHandler(FileSystemEventHandler):
    """Event handler for git directory changes."""

    def __init__(
        self,
        watcher: GitWatcher,
        event_queue: queue.Queue[Path],
        git_dir: Path,
    ):
        super().__init__()
        self._watcher = watcher
        self._queue = event_queue
        self._git_dir = git_dir

    def _queue_event(self) -> None:
        """Queue event with backpressure handling."""
        try:
            self._queue.put_nowait(self._git_dir)
        except queue.Full:
            self._watcher._dropped_events += 1
            if self._watcher._dropped_events % self._watcher._dropped_log_threshold == 0:
                logger.warning(
                    "Git watcher queue full, %d events dropped for repo %s",
                    self._watcher._dropped_events,
                    self._git_dir.parent,
                )

    def on_modified(self, event: FileSystemEvent) -> None:
        """Detect commits via refs/ or HEAD changes."""
        if event.is_directory:
            return

        path = Path(str(event.src_path))

        # Trigger on HEAD or refs/* changes
        if path.name == "HEAD" or "refs" in path.parts:
            self._queue_event()

    def on_created(self, event: FileSystemEvent) -> None:
        """Detect new branches/tags."""
        if not event.is_directory and "refs" in Path(str(event.src_path)).parts:
            self._queue_event()
