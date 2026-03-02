"""
Unit tests for LeaderElection and LifecycleCoordinator leader integration.

Commit 2.2: Verifies SQLite-based leader election ensures only one primary per project.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, cast

import pytest

from src.lifecycle import LeaderElection, LifecycleCoordinator, LifecycleState
from src.storage.db import DatabaseManager


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def db(tmp_path) -> DatabaseManager:
    """Provide a fresh DatabaseManager backed by a temp SQLite file."""
    return DatabaseManager(tmp_path / "test.db")


# ---------------------------------------------------------------------------
# Lightweight stubs (same pattern as test_lifecycle.py)
# ---------------------------------------------------------------------------


@dataclass
class FakeGitConfig:
    enabled: bool = False
    watch_enabled: bool = False


@dataclass
class FakeIndexingConfig:
    documents_path: str = "/tmp"
    exclude: list[str] = field(default_factory=list)
    exclude_hidden_dirs: bool = True


@dataclass
class FakeConfig:
    git_indexing: FakeGitConfig = field(default_factory=FakeGitConfig)
    indexing: FakeIndexingConfig = field(default_factory=FakeIndexingConfig)


@dataclass
class FakeContext:
    """Minimal stub to satisfy LifecycleCoordinator.start()."""

    config: FakeConfig = field(default_factory=FakeConfig)
    commit_indexer: None = None

    async def start(self, background_index: bool = False) -> None:
        pass

    async def stop(self) -> None:
        pass

    async def ensure_ready(self, timeout: float = 60.0) -> None:
        pass


def _fake_ctx() -> Any:
    return cast(Any, FakeContext())


# ---------------------------------------------------------------------------
# LeaderElection tests
# ---------------------------------------------------------------------------


class TestLeaderElection:
    def test_first_instance_becomes_leader(self, db: DatabaseManager) -> None:
        """First instance to try_acquire becomes leader."""
        election = LeaderElection(db, instance_id="instance-1")
        assert election.try_acquire() is True
        assert election.is_leader is True

    def test_second_instance_becomes_replica(self, db: DatabaseManager) -> None:
        """Second instance cannot acquire while first is leader."""
        first = LeaderElection(db, instance_id="instance-1")
        assert first.try_acquire() is True

        second = LeaderElection(db, instance_id="instance-2")
        assert second.try_acquire() is False
        assert second.is_leader is False

    def test_leader_handoff_after_timeout(self, db: DatabaseManager) -> None:
        """After leader timeout, another instance can acquire."""
        first = LeaderElection(db, instance_id="instance-1")
        first._leader_timeout = 1.0  # short timeout for testing
        assert first.try_acquire() is True

        # Manually set heartbeat to the past (beyond the timeout)
        conn = db.get_connection()
        stale_data = json.dumps({
            "instance_id": "instance-1",
            "heartbeat": time.time() - 10.0,  # 10s ago
            "acquired_at": time.time() - 10.0,
        })
        conn.execute(
            "INSERT OR REPLACE INTO system_state (key, value) VALUES (?, ?)",
            ("leader_id", stale_data),
        )
        conn.commit()

        second = LeaderElection(db, instance_id="instance-2")
        second._leader_timeout = 1.0
        assert second.try_acquire() is True
        assert second.is_leader is True

    def test_heartbeat_keeps_leadership(self, db: DatabaseManager) -> None:
        """Regular heartbeats prevent timeout."""
        first = LeaderElection(db, instance_id="instance-1")
        assert first.try_acquire() is True

        # Heartbeat refreshes the timestamp
        first.heartbeat()

        second = LeaderElection(db, instance_id="instance-2")
        assert second.try_acquire() is False

    def test_release_allows_takeover(self, db: DatabaseManager) -> None:
        """After release(), another instance can acquire immediately."""
        first = LeaderElection(db, instance_id="instance-1")
        assert first.try_acquire() is True

        first.release()
        assert first.is_leader is False

        second = LeaderElection(db, instance_id="instance-2")
        assert second.try_acquire() is True
        assert second.is_leader is True

    def test_leader_release_is_idempotent(self, db: DatabaseManager) -> None:
        """Releasing when not leader is a no-op."""
        election = LeaderElection(db, instance_id="instance-1")
        # Never acquired — release should be fine
        election.release()
        assert election.is_leader is False

        # Acquire, release, release again
        assert election.try_acquire() is True
        election.release()
        election.release()
        assert election.is_leader is False


# ---------------------------------------------------------------------------
# LifecycleCoordinator + LeaderElection integration tests
# ---------------------------------------------------------------------------


class TestLifecycleLeaderElection:
    @pytest.mark.asyncio
    async def test_start_with_db_becomes_primary(self, db: DatabaseManager) -> None:
        """start() with db_manager results in READY_PRIMARY."""
        coord = LifecycleCoordinator()
        await coord.start(_fake_ctx(), db_manager=db)
        assert coord.state == LifecycleState.READY_PRIMARY
        await coord.shutdown()

    @pytest.mark.asyncio
    async def test_second_instance_becomes_replica(self, db: DatabaseManager) -> None:
        """Second coordinator with same db becomes READY_REPLICA."""
        coord1 = LifecycleCoordinator()
        await coord1.start(_fake_ctx(), db_manager=db)
        assert coord1.state == LifecycleState.READY_PRIMARY

        coord2 = LifecycleCoordinator()
        await coord2.start(_fake_ctx(), db_manager=db)
        assert coord2.state == LifecycleState.READY_REPLICA

        await coord1.shutdown()
        await coord2.shutdown()

    @pytest.mark.asyncio
    async def test_shutdown_releases_leader(self, db: DatabaseManager) -> None:
        """shutdown() releases the leader lock."""
        coord1 = LifecycleCoordinator()
        await coord1.start(_fake_ctx(), db_manager=db)
        assert coord1.state == LifecycleState.READY_PRIMARY

        await coord1.shutdown()

        # Now a new coordinator can become primary
        coord2 = LifecycleCoordinator()
        await coord2.start(_fake_ctx(), db_manager=db)
        assert coord2.state == LifecycleState.READY_PRIMARY
        await coord2.shutdown()

    @pytest.mark.asyncio
    async def test_start_without_db_gives_ready(self) -> None:
        """start() without db_manager gives plain READY (backward compatible)."""
        coord = LifecycleCoordinator()
        await coord.start(_fake_ctx())
        assert coord.state == LifecycleState.READY
        await coord.shutdown()
