from __future__ import annotations

import json
import os
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from searchkernel.api import DatabaseManager


class LeaderElection:
    """SQLite-backed leader election for shared daemon runtimes."""

    def __init__(
        self,
        db_manager: DatabaseManager,
        instance_id: str | None = None,
    ) -> None:
        self._db = db_manager
        self._instance_id = instance_id or f"pid_{os.getpid()}_{time.monotonic_ns()}"
        self._heartbeat_interval = 5.0
        self._leader_timeout = 15.0
        self._is_leader = False

    @property
    def is_leader(self) -> bool:
        return self._is_leader

    @property
    def instance_id(self) -> str:
        return self._instance_id

    def try_acquire(self) -> bool:
        conn = self._db.get_connection()
        now = time.time()
        try:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT value FROM system_state WHERE key = 'leader_id'"
            ).fetchone()
            if row is not None:
                leader_data = json.loads(row[0])
                if now - leader_data.get("heartbeat", 0) < self._leader_timeout:
                    if leader_data.get("instance_id") == self._instance_id:
                        conn.rollback()
                        self._is_leader = True
                        return True
                    conn.rollback()
                    self._is_leader = False
                    return False

            conn.execute(
                "INSERT OR REPLACE INTO system_state (key, value) VALUES (?, ?)",
                (
                    "leader_id",
                    json.dumps(
                        {
                            "instance_id": self._instance_id,
                            "heartbeat": now,
                            "acquired_at": now,
                        }
                    ),
                ),
            )
            conn.commit()
            self._is_leader = True
            return True
        except Exception:
            conn.rollback()
            raise

    def release(self) -> None:
        if not self._is_leader:
            return
        conn = self._db.get_connection()
        row = conn.execute(
            "SELECT value FROM system_state WHERE key = 'leader_id'"
        ).fetchone()
        if row:
            leader_data = json.loads(row[0])
            if leader_data.get("instance_id") == self._instance_id:
                conn.execute(
                    "DELETE FROM system_state WHERE key = ? AND value = ?",
                    ("leader_id", row[0]),
                )
                conn.commit()
        self._is_leader = False

    def heartbeat(self) -> bool:
        if not self._is_leader:
            return False
        conn = self._db.get_connection()
        try:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT value FROM system_state WHERE key = 'leader_id'"
            ).fetchone()
            if row is None:
                conn.rollback()
                self._is_leader = False
                return False

            leader_data = json.loads(row[0])
            if leader_data.get("instance_id") != self._instance_id:
                conn.rollback()
                self._is_leader = False
                return False

            leader_data["heartbeat"] = time.time()
            conn.execute(
                "UPDATE system_state SET value = ? WHERE key = ? AND value = ?",
                (json.dumps(leader_data), "leader_id", row[0]),
            )
            conn.commit()
            self._is_leader = True
            return True
        except Exception:
            conn.rollback()
            raise
