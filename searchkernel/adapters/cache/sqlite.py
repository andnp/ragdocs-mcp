"""SQLite-backed cache store with epoch-based invalidation."""

import json
import logging
import sqlite3
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class SQLiteCacheStore:
    """SQLite-backed cache store implementing the CacheStore port.

    Persists key-value pairs to a SQLite database. Entries are tagged with
    an epoch for invalidation support. Designed for durability across restarts.
    """

    def __init__(self, db_path: Path | str):
        """Initialize SQLite cache store.

        Args:
            db_path: Path to SQLite database file.
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _init_schema(self) -> None:
        """Create the cache table if it doesn't exist."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cache_store (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    epoch INTEGER NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_cache_epoch
                ON cache_store (epoch);
            """)
            conn.commit()

    def get(self, key: str) -> Any | None:
        """Retrieve a cached value.

        Args:
            key: Cache key.

        Returns:
            Cached value, or None if not found.
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    "SELECT value FROM cache_store WHERE key = ?;",
                    (key,),
                )
                row = cursor.fetchone()
                if row:
                    value_json = row[0]
                    return json.loads(value_json)
                return None
        except (sqlite3.DatabaseError, json.JSONDecodeError) as e:
            logger.warning(f"Error retrieving cache key {key}: {e}", exc_info=True)
            return None

    def set(self, key: str, value: Any, epoch: int) -> None:
        """Store a value with an associated epoch.

        Args:
            key: Cache key.
            value: Value to cache.
            epoch: Index epoch at cache time. Used for invalidation.
        """
        try:
            value_json = json.dumps(value)
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    INSERT INTO cache_store (key, value, epoch)
                    VALUES (?, ?, ?)
                    ON CONFLICT (key) DO UPDATE SET
                        value = EXCLUDED.value,
                        epoch = EXCLUDED.epoch;
                    """,
                    (key, value_json, epoch),
                )
                conn.commit()
        except (sqlite3.DatabaseError, TypeError) as e:
            logger.error(f"Error storing cache key {key}: {e}", exc_info=True)

    def invalidate_epoch(self, epoch: int) -> None:
        """Invalidate all entries from an epoch or earlier.

        Args:
            epoch: Entries with epoch <= this are discarded.
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    "DELETE FROM cache_store WHERE epoch <= ?;",
                    (epoch,),
                )
                conn.commit()
                logger.debug(
                    f"Invalidated cache entries for epochs <= {epoch} "
                    f"({cursor.rowcount} entries deleted)"
                )
        except sqlite3.DatabaseError as e:
            logger.error(f"Error invalidating cache epoch {epoch}: {e}", exc_info=True)
