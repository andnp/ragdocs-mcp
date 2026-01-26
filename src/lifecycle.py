from __future__ import annotations

import asyncio
import logging
import multiprocessing
import multiprocessing.synchronize
import os
import signal
import sys
import threading
import time
from dataclasses import dataclass, field
from enum import StrEnum
from multiprocessing import Queue
from pathlib import Path
from typing import TYPE_CHECKING, Any

from src.ipc.commands import (
    HealthCheckCommand,
    HealthStatusResponse,
    InitCompleteNotification,
    ShutdownCommand,
)

if TYPE_CHECKING:
    from src.context import ApplicationContext
    from src.git.watcher import GitWatcher
    from src.reader.context import ReadOnlyContext

logger = logging.getLogger(__name__)


class LifecycleState(StrEnum):
    UNINITIALIZED = "uninitialized"
    STARTING = "starting"
    INITIALIZING = "initializing"
    READY = "ready"
    DEGRADED = "degraded"
    SHUTTING_DOWN = "shutting_down"
    TERMINATED = "terminated"


@dataclass
class LifecycleCoordinator:
    _state: LifecycleState = field(default=LifecycleState.UNINITIALIZED)
    _ctx: ApplicationContext | None = field(default=None)
    _readonly_ctx: ReadOnlyContext | None = field(default=None)
    _git_watcher: GitWatcher | None = field(default=None, repr=False)
    _emergency_timer: threading.Timer | None = field(default=None, repr=False)
    _shutdown_count: int = field(default=0, repr=False)
    _graceful_timeout: float = field(default=2.0, repr=False)
    _forced_timeout: float = field(default=1.0, repr=False)
    _emergency_timeout: float = field(default=3.5, repr=False)
    _worker_process: multiprocessing.Process | None = field(default=None, repr=False)
    _command_queue: Queue[Any] | None = field(default=None, repr=False)
    _response_queue: Queue[Any] | None = field(default=None, repr=False)
    _shutdown_event: multiprocessing.synchronize.Event | None = field(default=None, repr=False)
    _restart_attempts: int = field(default=0, repr=False)
    _health_check_task: asyncio.Task[None] | None = field(default=None, repr=False)

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

    async def start_with_worker(self, readonly_ctx: ReadOnlyContext) -> None:
        if self._state != LifecycleState.UNINITIALIZED:
            raise RuntimeError(f"Cannot start from state {self._state}")

        self._state = LifecycleState.STARTING
        self._readonly_ctx = readonly_ctx

        try:
            await self._start_worker()

            worker_config = readonly_ctx.config.worker
            init_received = await self._wait_for_init(worker_config.startup_timeout)

            if init_received:
                self._state = LifecycleState.READY
                logger.info("Lifecycle: READY (worker initialized)")
            else:
                self._state = LifecycleState.INITIALIZING
                logger.warning("Lifecycle: INITIALIZING (worker init timeout, continuing)")

            await readonly_ctx.start_sync_watcher()

            self._health_check_task = asyncio.create_task(self._health_check_loop())

        except Exception:
            logger.error("Worker startup failed", exc_info=True)
            await self._cleanup_resources()
            self._state = LifecycleState.TERMINATED
            raise

    async def _start_worker(self) -> None:
        if self._readonly_ctx is None:
            raise RuntimeError("No readonly context")

        config = self._readonly_ctx.config

        self._command_queue = multiprocessing.Queue()
        self._response_queue = multiprocessing.Queue()
        self._shutdown_event = multiprocessing.Event()

        snapshot_base = Path(config.indexing.index_path) / "snapshots"
        snapshot_base.mkdir(parents=True, exist_ok=True)

        config_dict = {
            "documents_path": config.indexing.documents_path,
            "index_path": config.indexing.index_path,
            "indexing": {
                "recursive": config.indexing.recursive,
                "include": config.indexing.include,
                "exclude": config.indexing.exclude,
                "exclude_hidden_dirs": config.indexing.exclude_hidden_dirs,
            },
        }

        from src.worker.process import worker_main

        self._worker_process = multiprocessing.Process(
            target=worker_main,
            args=(
                config_dict,
                self._command_queue,
                self._response_queue,
                self._shutdown_event,
                snapshot_base,
            ),
            daemon=True,
        )
        self._worker_process.start()
        logger.info("Worker process started (pid=%d)", self._worker_process.pid)

    async def _wait_for_init(self, timeout: float) -> bool:
        if self._response_queue is None:
            return False

        start = time.monotonic()

        while time.monotonic() - start < timeout:
            try:
                message = self._response_queue.get_nowait()
                if isinstance(message, InitCompleteNotification):
                    logger.info(
                        "Worker init complete: v%d, %d docs",
                        message.version,
                        message.doc_count,
                    )
                    return True
            except Exception:
                pass

            await asyncio.sleep(0.1)

        return False

    async def _health_check_loop(self) -> None:
        if self._readonly_ctx is None:
            return

        interval = self._readonly_ctx.config.worker.health_check_interval

        while True:
            try:
                await asyncio.sleep(interval)

                if not self._is_worker_alive():
                    logger.warning("Worker process died, attempting restart")
                    self._state = LifecycleState.DEGRADED
                    await self._restart_worker()
                    continue

                if self._command_queue is not None:
                    self._command_queue.put_nowait(HealthCheckCommand())

                    if self._response_queue is not None:
                        try:
                            response = await asyncio.wait_for(
                                asyncio.to_thread(self._response_queue.get, timeout=5.0),
                                timeout=6.0,
                            )
                            if isinstance(response, HealthStatusResponse):
                                if not response.healthy:
                                    logger.warning("Worker reports unhealthy")
                                    self._state = LifecycleState.DEGRADED
                                elif self._state == LifecycleState.DEGRADED:
                                    self._state = LifecycleState.READY
                        except asyncio.TimeoutError:
                            logger.warning("Health check response timeout")

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Health check error: %s", e, exc_info=True)

    def _is_worker_alive(self) -> bool:
        if self._worker_process is None:
            return False
        return self._worker_process.is_alive()

    async def _restart_worker(self) -> None:
        if self._readonly_ctx is None:
            return

        max_attempts = self._readonly_ctx.config.worker.max_restart_attempts
        backoff_base = self._readonly_ctx.config.worker.restart_backoff_base

        if self._restart_attempts >= max_attempts:
            logger.error("Max restart attempts reached, giving up")
            self._state = LifecycleState.DEGRADED
            return

        self._restart_attempts += 1
        backoff = backoff_base * (2 ** (self._restart_attempts - 1))
        logger.info("Restarting worker (attempt %d/%d) after %.1fs", self._restart_attempts, max_attempts, backoff)

        await asyncio.sleep(backoff)

        await self._stop_worker()
        await self._start_worker()

        worker_config = self._readonly_ctx.config.worker
        if await self._wait_for_init(worker_config.startup_timeout):
            self._state = LifecycleState.READY
            logger.info("Worker restart successful")
        else:
            logger.warning("Worker restart: init timeout")

    async def _stop_worker(self) -> None:
        if self._shutdown_event is not None:
            self._shutdown_event.set()

        if self._command_queue is not None:
            try:
                self._command_queue.put_nowait(ShutdownCommand(graceful=True))
            except Exception:
                pass

        if self._worker_process is not None and self._worker_process.is_alive():
            timeout = 5.0
            if self._readonly_ctx:
                timeout = self._readonly_ctx.config.worker.shutdown_timeout

            try:
                self._worker_process.join(timeout=timeout)
            except Exception:
                pass

            if self._worker_process.is_alive():
                logger.warning("Worker did not exit gracefully, terminating")
                self._worker_process.terminate()
                self._worker_process.join(timeout=1.0)

        self._worker_process = None

    async def wait_ready(self, timeout: float = 60.0) -> None:
        if self._state == LifecycleState.READY:
            return

        allowed_states = (
            LifecycleState.STARTING,
            LifecycleState.INITIALIZING,
            LifecycleState.DEGRADED,
        )
        if self._state not in allowed_states:
            raise RuntimeError(f"Cannot wait for ready from state {self._state}")

        start = time.monotonic()

        # Wait for STARTING to transition to INITIALIZING/READY first
        while self._state == LifecycleState.STARTING:
            if time.monotonic() - start > timeout:
                raise RuntimeError(f"Wait for ready timed out after {timeout}s (stuck in STARTING)")
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

        while time.monotonic() - start < timeout:
            if self._state == LifecycleState.READY:
                return
            if self._readonly_ctx and self._readonly_ctx.is_ready():
                self._state = LifecycleState.READY
                return
            await asyncio.sleep(0.1)

        raise RuntimeError(f"Wait for ready timed out after {timeout}s")

    def request_shutdown(self) -> None:
        self._shutdown_count += 1

        if self._shutdown_count >= 2:
            logger.warning("Forced exit (second signal)")
            self._force_exit()
            return

        if self._state == LifecycleState.SHUTTING_DOWN:
            return

        if self._state in (LifecycleState.READY, LifecycleState.DEGRADED, LifecycleState.INITIALIZING):
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

        if self._health_check_task is not None:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
            self._health_check_task = None

        await self._stop_worker()

        if self._git_watcher:
            try:
                await self._git_watcher.stop()
            except Exception as e:
                logger.error(f"Error stopping git watcher: {e}", exc_info=True)
            self._git_watcher = None

        if self._readonly_ctx:
            try:
                await self._readonly_ctx.stop()
            except Exception as e:
                logger.error(f"Error stopping readonly context: {e}", exc_info=True)

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
