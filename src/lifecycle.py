from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import sys
import threading
import time
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING

from src.context import ApplicationContext
from src.daemon import (
    DaemonMetadata,
    RuntimePaths,
    remove_daemon_metadata,
    write_daemon_metadata,
)
from src.git.watcher import GitWatcher

if TYPE_CHECKING:
    from src.storage.db import DatabaseManager
    from src.worker.consumer import HueyWorker

logger = logging.getLogger(__name__)


class LifecycleState(StrEnum):
    UNINITIALIZED = "uninitialized"
    STARTING = "starting"
    INITIALIZING = "initializing"
    READY = "ready"
    READY_PRIMARY = "ready_primary"
    READY_REPLICA = "ready_replica"
    SHUTTING_DOWN = "shutting_down"
    TERMINATED = "terminated"


class LeaderElection:
    """SQLite-based leader election using system_state table."""

    def __init__(
        self,
        db_manager: DatabaseManager,
        instance_id: str | None = None,
    ) -> None:
        self._db = db_manager
        self._instance_id = instance_id or f"pid_{os.getpid()}_{time.monotonic_ns()}"
        self._heartbeat_interval = 5.0  # seconds
        self._leader_timeout = 15.0  # seconds
        self._heartbeat_task: asyncio.Task[None] | None = None
        self._is_leader = False

    @property
    def is_leader(self) -> bool:
        return self._is_leader

    @property
    def instance_id(self) -> str:
        return self._instance_id

    def try_acquire(self) -> bool:
        """Try to become the leader. Returns True if acquired."""
        conn = self._db.get_connection()
        now = time.time()

        row = conn.execute(
            "SELECT value FROM system_state WHERE key = 'leader_id'"
        ).fetchone()

        if row is not None:
            leader_data = json.loads(row[0])
            last_heartbeat = leader_data.get("heartbeat", 0)

            # Leader is still alive — can't take over
            if now - last_heartbeat < self._leader_timeout:
                if leader_data.get("instance_id") == self._instance_id:
                    self._is_leader = True
                    return True
                return False

        # No leader or leader timed out — try to acquire
        leader_data_json = json.dumps(
            {
                "instance_id": self._instance_id,
                "heartbeat": now,
                "acquired_at": now,
            }
        )
        conn.execute(
            "INSERT OR REPLACE INTO system_state (key, value) VALUES (?, ?)",
            ("leader_id", leader_data_json),
        )
        conn.commit()
        self._is_leader = True
        return True

    def release(self) -> None:
        """Release leadership."""
        if not self._is_leader:
            return
        conn = self._db.get_connection()
        row = conn.execute(
            "SELECT value FROM system_state WHERE key = 'leader_id'"
        ).fetchone()
        if row:
            leader_data = json.loads(row[0])
            if leader_data.get("instance_id") == self._instance_id:
                conn.execute("DELETE FROM system_state WHERE key = 'leader_id'")
                conn.commit()
        self._is_leader = False

    def heartbeat(self) -> None:
        """Update heartbeat timestamp."""
        if not self._is_leader:
            return
        conn = self._db.get_connection()
        now = time.time()
        leader_data_json = json.dumps(
            {
                "instance_id": self._instance_id,
                "heartbeat": now,
            }
        )
        conn.execute(
            "INSERT OR REPLACE INTO system_state (key, value) VALUES (?, ?)",
            ("leader_id", leader_data_json),
        )
        conn.commit()


@dataclass
class LifecycleCoordinator:
    _state: LifecycleState = field(default=LifecycleState.UNINITIALIZED)
    _manage_daemon_metadata: bool = field(default=True, repr=False)
    _ctx: ApplicationContext | None = field(default=None)
    _git_watcher: GitWatcher | None = field(default=None, repr=False)
    _emergency_timer: threading.Timer | None = field(default=None, repr=False)
    _shutdown_count: int = field(default=0, repr=False)
    _graceful_timeout: float = field(default=2.0, repr=False)
    _forced_timeout: float = field(default=1.0, repr=False)
    _emergency_timeout: float = field(default=3.5, repr=False)
    _init_error: BaseException | None = field(default=None, repr=False)
    _leader_election: LeaderElection | None = field(default=None, repr=False)
    _huey_worker: HueyWorker | None = field(default=None, repr=False)
    _runtime_paths: RuntimePaths = field(
        default_factory=RuntimePaths.resolve, repr=False
    )
    _started_at: float | None = field(default=None, repr=False)

    @property
    def state(self) -> LifecycleState:
        return self._state

    def record_init_error(self, error: BaseException) -> None:
        """Record an initialization failure so waiting handlers fail fast."""
        self._init_error = error
        logger.error("Initialization failed: %s", error)

    async def start(
        self,
        ctx: ApplicationContext,
        *,
        background_index: bool = False,
        db_manager: DatabaseManager | None = None,
        huey_worker: HueyWorker | None = None,
    ) -> None:
        if self._state != LifecycleState.UNINITIALIZED:
            raise RuntimeError(f"Cannot start from state {self._state}")

        self._state = LifecycleState.STARTING
        self._ctx = ctx
        if self._started_at is None:
            self._started_at = time.time()
        self._write_daemon_metadata()

        try:
            await ctx.start(background_index=background_index)

            if db_manager is not None:
                self._leader_election = LeaderElection(db_manager)
                if self._leader_election.try_acquire():
                    logger.info("Lifecycle: leader elected")
                    if huey_worker is not None:
                        self._huey_worker = huey_worker
                        self._huey_worker.start()
                else:
                    logger.info(
                        "Lifecycle: replica mode (another instance is primary)"
                    )

            if (
                ctx.config.git_indexing.enabled
                and ctx.config.git_indexing.watch_enabled
            ):
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
                            poll_interval=ctx.config.git_indexing.poll_interval_seconds,
                            use_tasks=huey_worker is not None,
                        )
                        self._git_watcher.start()
                        logger.info(
                            f"Git watcher started for {len(repos)} repositories"
                        )

            if background_index:
                self._state = LifecycleState.INITIALIZING
                self._write_daemon_metadata()
                logger.info("Lifecycle: INITIALIZING (indices loading in background)")
            elif db_manager is not None:
                if self._leader_election is not None and self._leader_election.is_leader:
                    self._state = LifecycleState.READY_PRIMARY
                    self._write_daemon_metadata()
                    logger.info("Lifecycle: READY_PRIMARY (leader elected)")
                else:
                    self._state = LifecycleState.READY_REPLICA
                    self._write_daemon_metadata()
                    logger.info(
                        "Lifecycle: READY_REPLICA (another instance is primary)"
                    )
            else:
                self._state = LifecycleState.READY
                self._write_daemon_metadata()
                logger.info("Lifecycle: READY")
        except Exception:
            logger.error("Startup failed, cleaning up resources", exc_info=True)
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
        write_daemon_metadata(self._runtime_paths.metadata_path, metadata)

    def _remove_daemon_metadata(self) -> None:
        if not self._manage_daemon_metadata:
            return
        remove_daemon_metadata(self._runtime_paths.metadata_path)

    async def _cleanup_resources(self) -> None:
        self._cancel_emergency_timer()

        if self._huey_worker is not None:
            try:
                self._huey_worker.stop()
            except Exception as e:
                logger.error(f"Error stopping Huey worker: {e}", exc_info=True)
            self._huey_worker = None

        if self._leader_election is not None:
            try:
                self._leader_election.release()
            except Exception as e:
                logger.error(f"Error releasing leader lock: {e}", exc_info=True)
            self._leader_election = None

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
