from __future__ import annotations

import logging
import sqlite3
import threading
from pathlib import Path

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Thread-safe SQLite database manager with WAL mode."""

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        self._local = threading.local()
        db_path.parent.mkdir(parents=True, exist_ok=True)
        # Initialize schema via a temporary connection
        conn = self._open_connection()
        self._initialize_schema_on(conn)
        conn.close()

    def _open_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.row_factory = sqlite3.Row
        return conn

    def get_connection(self) -> sqlite3.Connection:
        """Return a per-thread SQLite connection."""
        conn = getattr(self._local, "connection", None)
        if conn is None:
            conn = self._open_connection()
            self._local.connection = conn
        return conn

    def _initialize_schema_on(self, conn: sqlite3.Connection) -> None:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS documents (
                doc_id TEXT PRIMARY KEY,
                file_path TEXT,
                content_hash TEXT,
                mtime REAL,
                status TEXT,
                indexed_at REAL
            );

            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id TEXT PRIMARY KEY,
                doc_id TEXT,
                content TEXT,
                metadata TEXT,
                vector BLOB,
                indexed_at REAL
            );

            CREATE TABLE IF NOT EXISTS kv_store (
                key TEXT PRIMARY KEY,
                value TEXT
            );

            CREATE VIRTUAL TABLE IF NOT EXISTS search_index USING fts5(
                chunk_id UNINDEXED,
                doc_id UNINDEXED,
                content,
                title,
                headers,
                tags,
                source_file UNINDEXED
            );
        """)
        conn.commit()

    def initialize_schema(self) -> None:
        self._initialize_schema_on(self.get_connection())

    def close(self) -> None:
        conn = getattr(self._local, "connection", None)
        if conn is not None:
            conn.close()
            self._local.connection = None
