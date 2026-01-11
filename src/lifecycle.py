from __future__ import annotations

import asyncio
import logging
import os
import signal
import sys
import threading
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.context import ApplicationContext
    from src.git.watcher import GitWatcher

logger = logging.getLogger(__name__)


class LifecycleState(StrEnum):
    UNINITIALIZED = "uninitialized"
    STARTING = "starting"
    INITIALIZING = "initializing"  # MCP running, indices loading in background
    READY = "ready"
    DEGRADED = "degraded"
    SHUTTING_DOWN = "shutting_down"
    TERMINATED = "terminated"


@dataclass
class LifecycleCoordinator:
    _state: LifecycleState = field(default=LifecycleState.UNINITIALIZED)
    _ctx: ApplicationContext | None = field(default=None)
    _git_watcher: GitWatcher | None = field(default=None, repr=False)
    _emergency_timer: threading.Timer | None = field(default=None, repr=False)
    _shutdown_count: int = field(default=0, repr=False)
    _graceful_timeout: float = field(default=2.0, repr=False)
    _forced_timeout: float = field(default=1.0, repr=False)
    _emergency_timeout: float = field(default=3.5, repr=False)

    @property
    def state(self) -> LifecycleState:
        return self._state

    def is_running(self) -> bool:
        return self._state in (LifecycleState.INITIALIZING, LifecycleState.READY, LifecycleState.DEGRADED)

    async def start(self, ctx: ApplicationContext, *, background_index: bool = False) -> None:
        if self._state != LifecycleState.UNINITIALIZED:
            raise RuntimeError(f"Cannot start from state {self._state}")

        self._state = LifecycleState.STARTING
        self._ctx = ctx

        try:
            await ctx.start(background_index=background_index)

            # Start git watcher if enabled
            if ctx.config.git_indexing.enabled and ctx.config.git_indexing.watch_enabled:
                if ctx.commit_indexer is not None:
                    from pathlib import Path
                    from src.git.repository import discover_git_repositories
                    from src.git.watcher import GitWatcher

                    repos = discover_git_repositories(
                        Path(ctx.config.indexing.documents_path),
                        ctx.config.indexing.exclude,
                        ctx.config.indexing.exclude_hidden_dirs,
                    )

                    if repos:
                        self._git_watcher = GitWatcher(
                            git_repos=repos,
                            commit_indexer=ctx.commit_indexer,
                            config=ctx.config,
                            cooldown=ctx.config.git_indexing.watch_cooldown,
                        )
                        self._git_watcher.start()
                        logger.info(f"Git watcher started for {len(repos)} repositories")

            if background_index:
                self._state = LifecycleState.INITIALIZING
                logger.info("Lifecycle: INITIALIZING (indices loading in background)")
            else:
                self._state = LifecycleState.READY
                logger.info("Lifecycle: READY")
        except Exception:
            self._state = LifecycleState.TERMINATED
            raise

    async def wait_ready(self, timeout: float = 60.0) -> None:
        """Wait for indices to be ready. Call from tool handlers before queries."""
        if self._state == LifecycleState.READY:
            return
        if self._state != LifecycleState.INITIALIZING:
            raise RuntimeError(f"Cannot wait for ready from state {self._state}")
        if self._ctx is None:
            raise RuntimeError("No context available")

        await self._ctx.ensure_ready(timeout=timeout)
        self._state = LifecycleState.READY
        logger.info("Lifecycle: READY (initialization complete)")

    def request_shutdown(self) -> None:
        self._shutdown_count += 1

        if self._shutdown_count >= 2:
            logger.warning("Forced exit (second signal)")
            self._force_exit()
            return

        if self._state == LifecycleState.SHUTTING_DOWN:
            return

        if self._state in (LifecycleState.READY, LifecycleState.DEGRADED):
            self._state = LifecycleState.SHUTTING_DOWN
            logger.info("Lifecycle: SHUTTING_DOWN")
            self._start_emergency_timer()
            self._close_stdin()

    def _close_stdin(self) -> None:
        try:
            sys.stdin.close()
        except Exception:
            pass
        try:
            os.close(0)
        except Exception:
            pass

    def _start_emergency_timer(self) -> None:
        def emergency_exit():
            logger.error(f"Emergency exit after {self._emergency_timeout}s")
            os._exit(1)

        self._emergency_timer = threading.Timer(
            self._emergency_timeout,
            emergency_exit,
        )
        self._emergency_timer.daemon = True
        self._emergency_timer.start()

    def _force_exit(self) -> None:
        os._exit(0)

    def _cancel_emergency_timer(self) -> None:
        if self._emergency_timer:
            self._emergency_timer.cancel()
            self._emergency_timer = None

    async def shutdown(self) -> None:
        if self._state == LifecycleState.TERMINATED:
            return

        self._state = LifecycleState.SHUTTING_DOWN

        # Stop git watcher first
        if self._git_watcher:
            try:
                await self._git_watcher.stop()
            except Exception as e:
                logger.error(f"Error stopping git watcher: {e}")
            self._git_watcher = None

        if self._ctx:
            try:
                async with asyncio.timeout(self._graceful_timeout):
                    await self._ctx.stop()
            except asyncio.TimeoutError:
                logger.warning("Graceful shutdown timed out")
            except Exception as e:
                logger.error(f"Error during shutdown: {e}")

        self._state = LifecycleState.TERMINATED
        self._cancel_emergency_timer()
        logger.info("Lifecycle: TERMINATED")

    def install_signal_handlers(self, loop: asyncio.AbstractEventLoop) -> None:
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, self.request_shutdown)
