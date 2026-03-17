"""Git polling watcher for automatic commit indexing.

Uses asyncio polling instead of inotify file-system listeners to avoid
consuming inode watches. Each poll queries git log for new commits since
the last indexed timestamp.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from src.config import Config
from src.git.commit_indexer import CommitIndexer

logger = logging.getLogger(__name__)


class GitWatcher:
    """Polls .git directories on a fixed interval and triggers incremental commit indexing."""

    def __init__(
        self,
        git_repos: list[Path],
        commit_indexer: CommitIndexer,
        config: Config,
        poll_interval: float = 30.0,
        use_tasks: bool = False,
    ):
        self._git_repos = git_repos
        self._commit_indexer = commit_indexer
        self._config = config
        self._poll_interval = poll_interval
        self._use_tasks = use_tasks
        self._running = False
        self._task: asyncio.Task[None] | None = None

    def start(self) -> None:
        """Start polling git directories."""
        if self._running:
            return

        self._running = True
        self._task = asyncio.create_task(self._poll_loop())
        logger.info(
            "Git poller started for %d repositories (poll interval: %ss)",
            len(self._git_repos),
            self._poll_interval,
        )

    async def stop(self) -> None:
        """Stop polling git directories."""
        if not self._running:
            return

        self._running = False

        if self._task:
            self._task.cancel()
            try:
                await asyncio.wait_for(self._task, timeout=1.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                pass
            self._task = None

        logger.info("Git poller stopped")

    async def _poll_loop(self) -> None:
        """Poll loop: sleep for the configured interval, then check all repos."""
        while self._running:
            try:
                await asyncio.sleep(self._poll_interval)
            except asyncio.CancelledError:
                break

            if self._running:
                await self._batch_process(set(self._git_repos))

    async def _batch_process(self, git_dirs: set[Path]) -> None:
        """Incrementally index any commits added since the last poll."""
        if self._use_tasks:
            from src.indexing.tasks import enqueue_refresh_git

            for git_dir in git_dirs:
                if enqueue_refresh_git(str(git_dir)):
                    logger.info("Enqueued git refresh task for %s", git_dir.parent)
                else:
                    logger.warning(
                        "Git watcher task mode enabled but no task was registered for %s",
                        git_dir.parent,
                    )
            return

        from src.git.parallel_indexer import (
            ParallelIndexingConfig,
            index_commits_parallel,
        )
        from src.git.repository import get_commits_after_timestamp

        parallel_config = ParallelIndexingConfig()

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
                    continue

                logger.info(
                    "Indexing %d new commits from %s",
                    len(commit_hashes),
                    git_dir.parent,
                )

                indexed = await index_commits_parallel(
                    commit_hashes,
                    git_dir,
                    self._commit_indexer,
                    parallel_config,
                    200,
                )

                logger.info(
                    "Updated commit index for %s: %d commits",
                    git_dir.parent.name,
                    indexed,
                )

            except Exception as e:
                logger.error(
                    "Failed to update commits for %s: %s", git_dir, e, exc_info=True
                )
