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
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from mcp_markdown_ragdocs.coordination.leader_election import LeaderElection
from mcp_markdown_ragdocs.coordination.lifecycle_ports import (
    DaemonMetadataAdapter,
    LifecycleMetadataPort,
    WorkerSupervisionAdapter,
    WorkerSupervisionPort,
)
from mcp_markdown_ragdocs.daemon import (
    DaemonMetadata,
    RuntimePaths,
)
from mcp_markdown_ragdocs.git.watcher import GitWatcher

if TYPE_CHECKING:
    from mcp_markdown_ragdocs.config import Config
    from mcp_markdown_ragdocs.indexing.record_manager import RecordIndexManager


logger = logging.getLogger(__name__)


class LifecycleContextPort(Protocol):
    """Application lifecycle operations required by the coordinator."""

    async def start(self, background_index: bool = False) -> None: ...

    async def stop(self) -> None: ...

    async def ensure_ready(self, timeout: float = 60.0) -> None: ...


@runtime_checkable
class GitIndexingContextPort(LifecycleContextPort, Protocol):
    """Optional context capabilities used to construct the Git watcher."""

    config: Config
    git_indexing_enabled: bool
    index_manager: RecordIndexManager

    def discover_git_repositories(self) -> list[Path]: ...


class LifecycleState(StrEnum):
    UNINITIALIZED = "uninitialized"
    STARTING = "starting"
    INITIALIZING = "initializing"
    READY = "ready"
    READY_PRIMARY = "ready_primary"
    READY_REPLICA = "ready_replica"
    SHUTTING_DOWN = "shutting_down"
    TERMINATED = "terminated"


@dataclass
class LifecycleCoordinator:
    _state: LifecycleState = field(default=LifecycleState.UNINITIALIZED)
    _manage_daemon_metadata: bool = field(default=True, repr=False)
    _ctx: LifecycleContextPort | None = field(default=None)
    _git_watcher: GitWatcher | None = field(default=None, repr=False)
    _emergency_timer: threading.Timer | None = field(default=None, repr=False)
    _shutdown_count: int = field(default=0, repr=False)
    _graceful_timeout: float = field(default=2.0, repr=False)
    _forced_timeout: float = field(default=1.0, repr=False)
    _emergency_timeout: float = field(default=3.5, repr=False)
    _init_error: BaseException | None = field(default=None, repr=False)
    _leader_election: Any = field(default=None, repr=False)
    _huey_worker: Any = field(default=None, repr=False)
    _runtime_paths: RuntimePaths = field(
        default_factory=RuntimePaths.resolve, repr=False
    )
    _started_at: float | None = field(default=None, repr=False)
    _readiness_task: asyncio.Task[None] | None = field(default=None, repr=False)
    _worker_supervision_task: asyncio.Task[None] | None = field(
        default=None,
        repr=False,
    )
    _leader_monitor_task: asyncio.Task[None] | None = field(
        default=None,
        repr=False,
    )
    _leader_heartbeat_task: asyncio.Task[None] | None = field(
        default=None,
        repr=False,
    )
    _leader_monitor_interval: float = field(default=5.0, repr=False)
    _metadata: LifecycleMetadataPort | None = field(default=None, repr=False)
    _worker_supervisor: WorkerSupervisionPort | None = field(default=None, repr=False)

    @property
    def state(self) -> LifecycleState:
        return self._state

    def record_init_error(self, error: BaseException) -> None:
        """Record an initialization failure so waiting handlers fail fast."""
        self._init_error = error
        logger.error("Initialization failed: %s", error)

    async def start(
        self,
        ctx: LifecycleContextPort,
        *,
        background_index: bool = False,
        db_manager: Any = None,
        huey_worker: Any = None,
    ) -> None:
        if self._state != LifecycleState.UNINITIALIZED:
            raise RuntimeError(f"Cannot start from state {self._state}")

        self._state = LifecycleState.STARTING
        self._ctx = ctx
        self._metadata = DaemonMetadataAdapter(self._runtime_paths)
        self._worker_supervisor = (
            WorkerSupervisionAdapter(huey_worker) if huey_worker is not None else None
        )
        if self._started_at is None:
            self._started_at = time.time()
        self._write_daemon_metadata()

        try:
            await ctx.start(background_index=background_index)

            if db_manager is not None:
                self._leader_election = LeaderElection(db_manager)
                self._huey_worker = huey_worker
                if await asyncio.to_thread(self._leader_election.try_acquire):
                    logger.info("Lifecycle: leader elected")
                    self._ensure_leader_heartbeat()
                    await self._ensure_huey_worker_running()
                else:
                    logger.info(
                        "Lifecycle: replica mode (another instance is primary)"
                    )

            git_context = ctx if isinstance(ctx, GitIndexingContextPort) else None
            if (
                git_context is not None
                and git_context.config.git_indexing.enabled
                and git_context.config.git_indexing.watch_enabled
                and huey_worker is None
            ) and git_context.git_indexing_enabled:
                repos = await asyncio.to_thread(
                    git_context.discover_git_repositories
                )

                if repos:
                    self._git_watcher = GitWatcher(
                        git_repos=repos,
                        index_manager=git_context.index_manager,
                        config=git_context.config,
                        poll_interval=git_context.config.git_indexing.poll_interval_seconds,
                        use_tasks=huey_worker is not None,
                    )
                    self._git_watcher.start()
                    logger.info(
                        f"Git watcher started for {len(repos)} repositories"
                    )

            if background_index:
                self._state = LifecycleState.INITIALIZING
                self._write_daemon_metadata()
                self._readiness_task = asyncio.create_task(
                    self._promote_when_ready()
                )
                self._ensure_leader_monitor()
                logger.info("Lifecycle: INITIALIZING (indices loading in background)")
            elif db_manager is not None:
                if self._leader_election is not None and self._leader_election.is_leader:
                    self._state = LifecycleState.READY_PRIMARY
                    self._write_daemon_metadata()
                    logger.info("Lifecycle: READY_PRIMARY (leader elected)")
                else:
                    self._state = LifecycleState.READY_REPLICA
                    self._write_daemon_metadata()
                    self._ensure_leader_monitor()
                    logger.info(
                        "Lifecycle: READY_REPLICA (another instance is primary)"
                    )
            else:
                self._state = LifecycleState.READY
                self._write_daemon_metadata()
                logger.info("Lifecycle: READY")
        except Exception:
            logger.exception("Startup failed, cleaning up resources")
            await self._cleanup_resources()
            self._state = LifecycleState.TERMINATED
            self._remove_daemon_metadata()
            raise

    async def wait_ready(self, timeout: float = 60.0) -> None:
        if self._state in (
            LifecycleState.READY,
            LifecycleState.READY_PRIMARY,
            LifecycleState.READY_REPLICA,
        ):
            return

        # Fail fast if initialization already failed
        if self._init_error is not None:
            raise RuntimeError(
                f"Server initialization failed: {self._init_error}"
            ) from self._init_error

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
                raise RuntimeError(
                    f"Server initialization failed: {self._init_error}"
                ) from self._init_error
            if time.monotonic() - start > timeout:
                raise RuntimeError(
                    f"Wait for ready timed out after {timeout}s (stuck in {self._state})"
                )
            await asyncio.sleep(0.1)

        if self._state in (
            LifecycleState.READY,
            LifecycleState.READY_PRIMARY,
            LifecycleState.READY_REPLICA,
        ):
            return

        if self._ctx is not None:
            remaining = timeout - (time.monotonic() - start)
            if remaining <= 0:
                raise RuntimeError(f"Wait for ready timed out after {timeout}s")
            await self._ctx.ensure_ready(timeout=remaining)
            if self._leader_election is not None:
                self._state = (
                    LifecycleState.READY_PRIMARY
                    if self._leader_election.is_leader
                    else LifecycleState.READY_REPLICA
                )
            else:
                self._state = LifecycleState.READY
            self._write_daemon_metadata()
            logger.info("Lifecycle: %s (initialization complete)", self._state)
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

        if self._state in (
            LifecycleState.READY,
            LifecycleState.READY_PRIMARY,
            LifecycleState.READY_REPLICA,
            LifecycleState.INITIALIZING,
            LifecycleState.STARTING,
        ):
            self._state = LifecycleState.SHUTTING_DOWN
            self._write_daemon_metadata()
            logger.info("Lifecycle: SHUTTING_DOWN")
            self._start_emergency_timer()
            self._close_stdin()

    def _close_stdin(self) -> None:
        try:
            sys.stdin.close()
        except OSError:
            logger.debug("Failed to close stdin stream", exc_info=True)
        try:
            os.close(0)
        except OSError:
            logger.debug("Failed to close stdin fd", exc_info=True)

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
        self._write_daemon_metadata()

        await self._cleanup_resources()

        self._state = LifecycleState.TERMINATED
        self._remove_daemon_metadata()
        self._cancel_emergency_timer()
        logger.info("Lifecycle: TERMINATED")

    def _write_daemon_metadata(self) -> None:
        if not self._manage_daemon_metadata:
            return
        if self._started_at is None:
            self._started_at = time.time()

        metadata = DaemonMetadata(
            pid=os.getpid(),
            started_at=self._started_at,
            status=self._state.value,
            socket_path=str(self._runtime_paths.socket_path),
            index_db_path=str(self._runtime_paths.index_db_path),
            queue_db_path=str(self._runtime_paths.queue_db_path),
        )
        if self._metadata is None:
            self._metadata = DaemonMetadataAdapter(self._runtime_paths)
        self._metadata.write(metadata)

    def _remove_daemon_metadata(self) -> None:
        if not self._manage_daemon_metadata:
            return
        if self._metadata is not None:
            self._metadata.remove()

    async def _promote_when_ready(self) -> None:
        if self._ctx is None:
            return

        try:
            await self._ctx.ensure_ready()
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001 -- startup boundary; must not crash the process
            self.record_init_error(e)
            return

        if self._state != LifecycleState.INITIALIZING:
            return

        if self._leader_election is not None:
            self._state = (
                LifecycleState.READY_PRIMARY
                if self._leader_election.is_leader
                else LifecycleState.READY_REPLICA
            )
        else:
            self._state = LifecycleState.READY

        self._write_daemon_metadata()
        if self._state == LifecycleState.READY_REPLICA:
            self._ensure_leader_monitor()
        logger.info("Lifecycle: %s (background initialization complete)", self._state)

    def _should_monitor_leader_failover(self) -> bool:
        leader_election = self._leader_election
        return (
            leader_election is not None
            and not leader_election.is_leader
            and self._state in (
                LifecycleState.INITIALIZING,
                LifecycleState.READY_REPLICA,
            )
        )

    def _ensure_leader_monitor(self) -> None:
        if not self._should_monitor_leader_failover():
            return
        if self._leader_monitor_task is not None and not self._leader_monitor_task.done():
            return
        self._leader_monitor_task = asyncio.create_task(self._monitor_leader_failover())

    def _ensure_leader_heartbeat(self) -> None:
        if self._leader_election is None or not self._leader_election.is_leader:
            return
        if (
            self._leader_heartbeat_task is not None
            and not self._leader_heartbeat_task.done()
        ):
            return
        self._leader_heartbeat_task = asyncio.create_task(self._heartbeat_leader())

    async def _ensure_huey_worker_running(self) -> None:
        if self._huey_worker is None:
            return
        if self._worker_supervisor is None:
            self._worker_supervisor = WorkerSupervisionAdapter(self._huey_worker)
        if self._worker_supervisor is None:
            return
        await self._worker_supervisor.start()
        if (
            self._worker_supervision_task is not None
            and not self._worker_supervision_task.done()
        ):
            return
        self._worker_supervision_task = asyncio.create_task(
            self._supervise_worker_health()
        )

    async def _monitor_leader_failover(self) -> None:
        while True:
            await asyncio.sleep(self._leader_monitor_interval)

            if self._state in (
                LifecycleState.SHUTTING_DOWN,
                LifecycleState.TERMINATED,
            ):
                return

            leader_election = self._leader_election
            if leader_election is None:
                return
            if not self._should_monitor_leader_failover():
                return
            if not await asyncio.to_thread(leader_election.try_acquire):
                continue

            self._ensure_leader_heartbeat()
            await self._ensure_huey_worker_running()

            if self._state == LifecycleState.READY_REPLICA:
                self._state = LifecycleState.READY_PRIMARY
                self._write_daemon_metadata()
                logger.info(
                    "Lifecycle: READY_PRIMARY (replica promoted after failover)"
                )
            else:
                self._write_daemon_metadata()
                logger.info(
                    "Lifecycle: INITIALIZING (replica acquired leadership after failover)"
                )
            return

    async def _heartbeat_leader(self) -> None:
        leader_election = self._leader_election
        if leader_election is None:
            return

        while True:
            await asyncio.sleep(getattr(leader_election, "_heartbeat_interval", 5.0))

            if self._state in (
                LifecycleState.SHUTTING_DOWN,
                LifecycleState.TERMINATED,
            ):
                return

            leader_election = self._leader_election
            if leader_election is None or not leader_election.is_leader:
                return

            try:
                if not await asyncio.to_thread(leader_election.heartbeat):
                    await self._step_down_after_leadership_loss()
                    return
            except Exception:
                logger.exception("Failed to refresh leader heartbeat")
                await self._step_down_after_leadership_loss()
                return

    async def _step_down_after_leadership_loss(self) -> None:
        leader_election = self._leader_election
        if leader_election is None or leader_election.is_leader:
            return

        if self._worker_supervision_task is not None:
            self._worker_supervision_task.cancel()
            await asyncio.gather(
                self._worker_supervision_task,
                return_exceptions=True,
            )
            self._worker_supervision_task = None

        if self._worker_supervisor is not None:
            try:
                await asyncio.to_thread(self._worker_supervisor.stop)
            except Exception:
                logger.exception("Error stopping worker after leadership loss")

        if self._state == LifecycleState.READY_PRIMARY:
            self._state = LifecycleState.READY_REPLICA
        self._write_daemon_metadata()
        self._ensure_leader_monitor()

    async def _cleanup_resources(self) -> None:
        self._cancel_emergency_timer()

        if self._leader_monitor_task is not None:
            self._leader_monitor_task.cancel()
            await asyncio.gather(self._leader_monitor_task, return_exceptions=True)
            self._leader_monitor_task = None

        if self._leader_heartbeat_task is not None:
            self._leader_heartbeat_task.cancel()
            await asyncio.gather(self._leader_heartbeat_task, return_exceptions=True)
            self._leader_heartbeat_task = None

        if self._worker_supervision_task is not None:
            self._worker_supervision_task.cancel()
            await asyncio.gather(self._worker_supervision_task, return_exceptions=True)
            self._worker_supervision_task = None

        if self._readiness_task is not None:
            self._readiness_task.cancel()
            await asyncio.gather(self._readiness_task, return_exceptions=True)
            self._readiness_task = None

        if self._huey_worker is not None:
            try:
                if self._worker_supervisor is not None:
                    self._worker_supervisor.stop()
            except Exception:
                logger.exception("Error stopping Huey worker")
            self._huey_worker = None
            self._worker_supervisor = None

        if self._leader_election is not None:
            try:
                await asyncio.to_thread(self._leader_election.release)
            except Exception:
                logger.exception("Error releasing leader lock")
            self._leader_election = None

        if self._git_watcher:
            try:
                await self._git_watcher.stop()
            except Exception:
                logger.exception("Error stopping git watcher")
            self._git_watcher = None

        if self._ctx:
            try:
                async with asyncio.timeout(self._graceful_timeout):
                    await self._ctx.stop()
            except TimeoutError:
                logger.warning("Graceful shutdown timed out")
            except Exception:
                logger.exception("Error during context cleanup")

    async def _supervise_worker_health(self) -> None:
        while True:
            await asyncio.sleep(5.0)

            worker = self._worker_supervisor
            if worker is None and self._huey_worker is not None:
                worker = WorkerSupervisionAdapter(self._huey_worker)
            if worker is None:
                return
            if self._state in (
                LifecycleState.SHUTTING_DOWN,
                LifecycleState.TERMINATED,
            ):
                return
            if self._leader_election is None or not self._leader_election.is_leader:
                return
            if worker.is_healthy():
                continue

            logger.warning("Huey worker subprocess is unhealthy; restarting")
            try:
                await worker.restart()
            except Exception:
                logger.exception("Failed to restart Huey worker subprocess")

    def install_signal_handlers(self, loop: asyncio.AbstractEventLoop) -> None:
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, self.request_shutdown)
