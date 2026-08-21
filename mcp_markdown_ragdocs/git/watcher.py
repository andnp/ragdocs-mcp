"""Git polling watcher for automatic commit indexing.

Uses asyncio polling instead of inotify file-system listeners to avoid
consuming inode watches. Each poll queries git log for new commits since
the last indexed timestamp.
"""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path

from mcp_markdown_ragdocs.config import Config, resolve_project_id_for_path
from mcp_markdown_ragdocs.git.repository import get_git_ref_signature
from mcp_markdown_ragdocs.coordination.task_submission import TaskSubmissionPort
from mcp_markdown_ragdocs.indexing.git_refresh_state import get_head, get_progress
from mcp_markdown_ragdocs.indexing.record_manager import RecordIndexManager as IndexManager

logger = logging.getLogger(__name__)


class GitWatcher:
    """Polls .git directories on a fixed interval and triggers incremental commit indexing."""

    def __init__(
        self,
        git_repos: list[Path],
        index_manager: IndexManager,
        config: Config,
        poll_interval: float = 300.0,
        use_tasks: bool = False,
        task_submission: TaskSubmissionPort | None = None,
    ):
        self._git_repos = git_repos
        self._index_manager = index_manager
        self._config = config
        self._poll_interval = poll_interval
        self._use_tasks = use_tasks
        self._task_submission = task_submission
        self._running = False
        self._task: asyncio.Task[None] | None = None
        self._last_indexed: dict[Path, int] = {}

    @property
    def _refresh_state_root(self) -> Path:
        return Path(self._config.indexing.index_path)

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
            except (TimeoutError, asyncio.CancelledError):
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
            if self._task_submission is None:
                raise RuntimeError(
                    "Task submission port is required when task mode is enabled"
                )
            submit_refresh = self._task_submission.submit_refresh_git_request

            signatures = await asyncio.gather(
                *(
                    asyncio.to_thread(get_git_ref_signature, git_dir)
                    for git_dir in git_dirs
                )
            )
            direct_refresh_dirs: set[Path] = set()
            for git_dir, signature in zip(git_dirs, signatures, strict=True):
                if signature is not None:
                    stored_head = get_head(self._refresh_state_root, git_dir)
                    progress = get_progress(self._refresh_state_root, git_dir)
                    completed_head = (
                        progress.get("observed_head")
                        if progress is not None
                        and progress.get("state") in {"completed", "skipped"}
                        else None
                    )
                    if stored_head == signature or completed_head == signature:
                        continue

                submission = submit_refresh(str(git_dir))
                if submission.enqueued:
                    logger.info("Enqueued git refresh task for %s", git_dir.parent)
                    continue
                if submission.status == "already_pending":
                    logger.info(
                        "Git refresh task already pending for %s",
                        git_dir.parent,
                    )
                    continue
                if submission.should_retry_later:
                    logger.warning(
                        "Skipping git refresh enqueue for %s due to task queue backpressure",
                        git_dir.parent,
                    )
                    continue

                logger.info(
                    "Git task queue unavailable for %s; falling back to direct refresh",
                    git_dir.parent,
                )
                direct_refresh_dirs.add(git_dir)

            if not direct_refresh_dirs:
                return
            git_dirs = direct_refresh_dirs

        from mcp_markdown_ragdocs.adapters.sources.git import GitContentSource
        from mcp_markdown_ragdocs.indexing.git_ingestion import (
            iter_git_ingestion_receipts,
        )
        from mcp_markdown_ragdocs.indexing.tasks import GIT_REFRESH_BATCH_SIZE

        for git_dir in git_dirs:
            try:
                poll_started_at = int(time.time())
                since = self._last_indexed.get(git_dir)

                source = GitContentSource(
                    git_dir,
                    workspace_id=resolve_project_id_for_path(
                        git_dir.parent,
                        self._config,
                    ),
                )
                indexed = 0
                async for receipt in iter_git_ingestion_receipts(
                    self._index_manager,
                    source,
                    since=str(since) if since is not None else None,
                    batch_size=GIT_REFRESH_BATCH_SIZE,
                ):
                    indexed += len(receipt.records)

                self._last_indexed[git_dir] = poll_started_at

                if indexed:
                    logger.info(
                        "Updated commit index for %s: %d commits",
                        git_dir.parent.name,
                        indexed,
                    )

            except Exception:
                logger.exception("Failed to update commits for %s", git_dir)
