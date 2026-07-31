"""File-watcher lifecycle management, extracted from ApplicationContext.

Owns the `FileWatcher` instance and the start/stop/refresh operations
`ApplicationContext` needs around it, so that config loading, index
bootstrap, and readiness state in `ApplicationContext` aren't tangled up
with watcher plumbing.

Note: git commit watching (`mcp_markdown_ragdocs.git.watcher.GitWatcher`) is
NOT owned by `ApplicationContext` -- it is only ever constructed by
`LifecycleCoordinator` (mcp_markdown_ragdocs/lifecycle.py) and ad hoc in
mcp_markdown_ragdocs/cli.py. `ApplicationContext` merely exposes a
`git_indexing_enabled` flag and some one-shot indexing helpers, so there is
no git-watcher lifecycle here to extract.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path

from mcp_markdown_ragdocs.indexing.manager import IndexManager
from mcp_markdown_ragdocs.indexing.watcher import FileWatcher

logger = logging.getLogger(__name__)


@dataclass
class WatcherLifecycle:
    """Owns the `FileWatcher` instance and its start/stop/refresh lifecycle."""

    watcher: FileWatcher | None = None

    @classmethod
    def create(
        cls,
        *,
        enabled: bool,
        documents_path: str,
        documents_roots: list[Path],
        index_manager: IndexManager,
        cooldown: float,
        include_patterns: list[str],
        exclude_patterns: list[str],
        exclude_hidden_dirs: bool,
        parser_suffixes: set[str],
        use_tasks: bool,
        task_backpressure_limit: int | None,
    ) -> WatcherLifecycle:
        if not enabled:
            return cls(watcher=None)

        watcher = FileWatcher(
            documents_path=documents_path,
            documents_paths=[str(root) for root in documents_roots],
            index_manager=index_manager,
            cooldown=cooldown,
            include_patterns=include_patterns,
            exclude_patterns=exclude_patterns,
            exclude_hidden_dirs=exclude_hidden_dirs,
            parser_suffixes=parser_suffixes,
            use_tasks=use_tasks,
            task_backpressure_limit=task_backpressure_limit,
        )
        return cls(watcher=watcher)

    def start(self, overflow_callback: Callable[[], Awaitable[None]]) -> None:
        """Wire the overflow callback and start the watcher, if enabled."""
        if self.watcher is None:
            return
        self.watcher.set_overflow_callback(overflow_callback)
        self.watcher.start()
        logger.info("File watcher started")

    def refresh_watches(self) -> None:
        """Pick up newly created watchable directories, if enabled."""
        if self.watcher is not None:
            self.watcher.refresh_watches()

    async def stop(self) -> None:
        """Stop the watcher with a bounded timeout, if enabled."""
        if self.watcher is None:
            return
        try:
            await asyncio.wait_for(self.watcher.stop(), timeout=1.0)
        except TimeoutError:
            logger.warning("FileWatcher stop timed out")
