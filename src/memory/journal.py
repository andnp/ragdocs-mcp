"""System 1 journal — raw thought capture and retrieval."""

from __future__ import annotations

import logging
import time

from src.storage.db import DatabaseManager

logger = logging.getLogger(__name__)


class JournalEntry:
    """A single System 1 journal entry."""

    __slots__ = ("id", "content", "timestamp", "status")

    def __init__(self, id: int, content: str, timestamp: float, status: str) -> None:
        self.id = id
        self.content = content
        self.timestamp = timestamp
        self.status = status

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "content": self.content,
            "timestamp": self.timestamp,
            "status": self.status,
        }


class System1Journal:
    """Manages raw thought capture in the system1_journal table.

    System 1 thoughts are unprocessed observations recorded during
    AI interactions. They accumulate as 'pending' entries until a
    consolidation task processes them into refined memories.
    """

    def __init__(self, db_manager: DatabaseManager) -> None:
        self._db = db_manager

    def record(self, content: str) -> JournalEntry:
        """Record a raw thought. Returns the created entry."""
        if not content or not content.strip():
            raise ValueError("Journal content cannot be empty")

        content = content.strip()
        now = time.time()
        conn = self._db.get_connection()
        cursor = conn.execute(
            "INSERT INTO system1_journal (content, timestamp, status) VALUES (?, ?, ?)",
            (content, now, "pending"),
        )
        conn.commit()
        entry_id = cursor.lastrowid
        assert entry_id is not None

        logger.info("Recorded journal entry %d (%d chars)", entry_id, len(content))
        return JournalEntry(id=entry_id, content=content, timestamp=now, status="pending")

    def get_pending(self, limit: int = 50) -> list[JournalEntry]:
        """Get pending entries, oldest first."""
        conn = self._db.get_connection()
        rows = conn.execute(
            "SELECT id, content, timestamp, status FROM system1_journal "
            "WHERE status = 'pending' ORDER BY timestamp ASC LIMIT ?",
            (limit,),
        ).fetchall()
        return [JournalEntry(id=r[0], content=r[1], timestamp=r[2], status=r[3]) for r in rows]

    def mark_processed(self, entry_ids: list[int]) -> int:
        """Mark entries as processed. Returns count updated."""
        if not entry_ids:
            return 0
        conn = self._db.get_connection()
        placeholders = ",".join("?" for _ in entry_ids)
        cursor = conn.execute(
            f"UPDATE system1_journal SET status = 'processed' "
            f"WHERE id IN ({placeholders}) AND status = 'pending'",
            entry_ids,
        )
        conn.commit()
        return cursor.rowcount

    def mark_archived(self, entry_ids: list[int]) -> int:
        """Move processed entries to archived. Returns count updated."""
        if not entry_ids:
            return 0
        conn = self._db.get_connection()
        placeholders = ",".join("?" for _ in entry_ids)
        cursor = conn.execute(
            f"UPDATE system1_journal SET status = 'archived' "
            f"WHERE id IN ({placeholders}) AND status = 'processed'",
            entry_ids,
        )
        conn.commit()
        return cursor.rowcount

    def count_by_status(self) -> dict[str, int]:
        """Count entries by status."""
        conn = self._db.get_connection()
        rows = conn.execute(
            "SELECT status, COUNT(*) FROM system1_journal GROUP BY status"
        ).fetchall()
        return {row[0]: row[1] for row in rows}

    def get_recent(self, limit: int = 10) -> list[JournalEntry]:
        """Get most recent entries regardless of status."""
        conn = self._db.get_connection()
        rows = conn.execute(
            "SELECT id, content, timestamp, status FROM system1_journal "
            "ORDER BY timestamp DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [JournalEntry(id=r[0], content=r[1], timestamp=r[2], status=r[3]) for r in rows]
