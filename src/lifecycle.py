from __future__ import annotations

import asyncio
import logging
import os
import signal
import sys
import threading
import time
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

from src.context import ApplicationContext
from src.git.watcher import GitWatcher

logger = logging.getLogger(__name__)


class LifecycleState(StrEnum):
    UNINITIALIZED = "uninitialized"
    STARTING = "starting"
    INITIALIZING = "initializing"
    READY = "ready"
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
    _init_error: BaseException | None = field(default=None, repr=False)

    @property
    def state(self) -> LifecycleState:
        return self._state

    def record_init_error(self, error: BaseException) -> None:
        """Record an initialization failure so waiting handlers fail fast."""
        self._init_error = error
        logger.error("Initialization failed: %s", error)

    async def start(self, ctx: ApplicationContext, *, background_index: bool = False) -> None:
        if self._state != LifecycleState.UNINITIALIZED:
            raise RuntimeError(f"Cannot start from state {self._state}")

        self._state = LifecycleState.STARTING
        self._ctx = ctx

        try:
            await ctx.start(background_index=background_index)

            if ctx.config.git_indexing.enabled and ctx.config.git_indexing.watch_enabled:
                if ctx.commit_indexer is not None:
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
            logger.error("Startup failed, cleaning up resources", exc_info=True)
            await self._cleanup_resources()
            self._state = LifecycleState.TERMINATED
            raise

    async def wait_ready(self, timeout: float = 60.0) -> None:
        if self._state == LifecycleState.READY:
            return

        # Fail fast if initialization already failed
        if self._init_error is not None:
            raise RuntimeError(f"Server initialization failed: {self._init_error}") from self._init_error

        allowed_states = (
            LifecycleState.UNINITIALIZED,
            LifecycleState.STARTING,
            LifecycleState.INITIALIZING,
        )
        if self._state not in allowed_states:
            raise RuntimeError(f"Cannot wait for ready from state {self._state}")

        start = time.monotonic()

        # Wait for UNINITIALIZED/STARTING to transition forward
        while self._state in (LifecycleState.UNINITIALIZED, LifecycleState.STARTING):
            if self._init_error is not None:
                raise RuntimeError(f"Server initialization failed: {self._init_error}") from self._init_error
            if time.monotonic() - start > timeout:
                raise RuntimeError(f"Wait for ready timed out after {timeout}s (stuck in {self._state})")
            await asyncio.sleep(0.1)

        if self._state == LifecycleState.READY:
            return

        if self._ctx is not None:
            remaining = timeout - (time.monotonic() - start)
            if remaining <= 0:
                raise RuntimeError(f"Wait for ready timed out after {timeout}s")
            await self._ctx.ensure_ready(timeout=remaining)
            self._state = LifecycleState.READY
            logger.info("Lifecycle: READY (initialization complete)")
            return

        raise RuntimeError(f"Wait for ready timed out after {timeout}s")

    def request_shutdown(self) -> None:
        self._shutdown_count += 1

        if self._shutdown_count >= 2:
            logger.warning("Forced exit (second signal)")
            self._force_exit()
            return

        if self._state == LifecycleState.SHUTTING_DOWN:
            return

        if self._state in (LifecycleState.READY, LifecycleState.INITIALIZING, LifecycleState.STARTING):
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

        await self._cleanup_resources()

        self._state = LifecycleState.TERMINATED
        self._cancel_emergency_timer()
        logger.info("Lifecycle: TERMINATED")

    async def _cleanup_resources(self) -> None:
        self._cancel_emergency_timer()

        if self._git_watcher:
            try:
                await self._git_watcher.stop()
            except Exception as e:
                logger.error(f"Error stopping git watcher: {e}", exc_info=True)
            self._git_watcher = None

        if self._ctx:
            try:
                async with asyncio.timeout(self._graceful_timeout):
                    await self._ctx.stop()
            except asyncio.TimeoutError:
                logger.warning("Graceful shutdown timed out")
            except Exception as e:
                logger.error(f"Error during context cleanup: {e}", exc_info=True)

    def install_signal_handlers(self, loop: asyncio.AbstractEventLoop) -> None:
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, self.request_shutdown)
