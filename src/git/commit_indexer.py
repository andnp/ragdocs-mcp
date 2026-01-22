"""SQLite-based commit index with embedding storage."""

import json
import logging
import sqlite3
import time
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from src.indices.vector import VectorIndex

logger = logging.getLogger(__name__)

SQLITE_CORRUPTION_PATTERNS = (
    "database disk image is malformed",
    "database is locked",
    "disk i/o error",
    "unable to open database file",
    "file is not a database",
)


class CommitIndexer:
    """Manages git commit index with embeddings."""

    def __init__(
        self,
        db_path: Path,
        embedding_model: VectorIndex,
    ):
        """
        Initialize commit indexer.

        Args:
            db_path: Path to SQLite database file
            embedding_model: VectorIndex for embedding generation
        """
        self._db_path = db_path
        self._embedding_model = embedding_model
        self._conn: sqlite3.Connection | None = None
        self._ensure_schema()

    @staticmethod
    def _normalize_repo_path(repo_path: str) -> str:
        """
        Normalize repository path for consistent storage and querying.

        Ensures:
        - Absolute path
        - No trailing slashes
        - No .git suffix

        Args:
            repo_path: Raw repository path

        Returns:
            Normalized absolute path string
        """
        path = Path(repo_path)

        # Strip .git suffix if present
        if path.name == ".git":
            path = path.parent

        # Resolve to absolute path and remove trailing slashes
        normalized = str(path.resolve())

        return normalized

    def _is_corruption_error(self, error: Exception) -> bool:
        error_msg = str(error).lower()
        return any(pattern in error_msg for pattern in SQLITE_CORRUPTION_PATTERNS)

    def _get_connection(self) -> sqlite3.Connection:
        if self._conn is None:
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                self._conn = sqlite3.connect(
                    str(self._db_path),
                    check_same_thread=False,
                    timeout=10.0,
                )
                self._conn.row_factory = sqlite3.Row
                self._conn.execute("PRAGMA journal_mode=WAL")
                self._conn.execute("PRAGMA synchronous=NORMAL")
                self._conn.execute("PRAGMA integrity_check(1)").fetchone()
            except sqlite3.DatabaseError as e:
                if self._is_corruption_error(e):
                    self._reinitialize_after_corruption()
                    return self._get_connection()
                raise
        return self._conn

    def _ensure_schema(self) -> None:
        """Create git_commits table if not exists."""
        conn = self._get_connection()

        conn.execute("""
            CREATE TABLE IF NOT EXISTS git_commits (
                hash TEXT PRIMARY KEY,
                timestamp INTEGER NOT NULL,
                author TEXT NOT NULL,
                committer TEXT NOT NULL,
                title TEXT NOT NULL,
                message TEXT NOT NULL,
                files_changed TEXT NOT NULL,
                delta_truncated TEXT NOT NULL,
                embedding BLOB NOT NULL,
                indexed_at INTEGER NOT NULL,
                repo_path TEXT NOT NULL
            )
        """)

        # Create indexes for efficient querying
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_timestamp
            ON git_commits(timestamp)
        """)

        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_indexed_at
            ON git_commits(indexed_at)
        """)

        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_repo_path
            ON git_commits(repo_path)
        """)

        conn.commit()
        logger.debug(f"Ensured schema for git_commits at {self._db_path}")

    def _reinitialize_after_corruption(self) -> None:
        logger.warning(
            f"SQLite database corruption detected at {self._db_path}. "
            "Recreating database (commits will be re-indexed)."
        )

        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None

        try:
            if self._db_path.exists():
                self._db_path.unlink()
            wal_path = self._db_path.with_suffix(".db-wal")
            shm_path = self._db_path.with_suffix(".db-shm")
            if wal_path.exists():
                wal_path.unlink()
            if shm_path.exists():
                shm_path.unlink()
        except OSError as e:
            logger.error(f"Failed to delete corrupted database: {e}", exc_info=True)
            raise RuntimeError(f"Cannot recover from database corruption: {e}") from e

        self._ensure_schema()
        logger.info(f"Database recreated at {self._db_path}")

    def add_commit(
        self,
        hash: str,
        timestamp: int,
        author: str,
        committer: str,
        title: str,
        message: str,
        files_changed: list[str],
        delta_truncated: str,
        commit_document: str,
        repo_path: str = "",
    ) -> None:
        try:
            embedding = self._embedding_model.get_text_embedding(commit_document)
            embedding_bytes = self._serialize_embedding(embedding)

            conn = self._get_connection()
            indexed_at = int(time.time())
            normalized_path = self._normalize_repo_path(repo_path)

            conn.execute(
                """
                INSERT OR REPLACE INTO git_commits
                (hash, timestamp, author, committer, title, message,
                 files_changed, delta_truncated, embedding, indexed_at, repo_path)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    hash,
                    timestamp,
                    author,
                    committer,
                    title,
                    message,
                    json.dumps(files_changed),
                    delta_truncated,
                    embedding_bytes,
                    indexed_at,
                    normalized_path,
                ),
            )
            conn.commit()
            logger.debug(f"Indexed commit {hash[:8]}")
        except sqlite3.DatabaseError as e:
            if self._is_corruption_error(e):
                self._reinitialize_after_corruption()
                self.add_commit(
                    hash, timestamp, author, committer, title, message,
                    files_changed, delta_truncated, commit_document, repo_path
                )
                return
            raise

    def remove_commit(self, commit_hash: str) -> None:
        """Remove commit from index by hash."""
        conn = self._get_connection()
        conn.execute("DELETE FROM git_commits WHERE hash = ?", (commit_hash,))
        conn.commit()
        logger.debug(f"Removed commit {commit_hash[:8]}")

    def clear(self) -> None:
        """Remove all commits from index."""
        conn = self._get_connection()
        conn.execute("DELETE FROM git_commits")
        conn.commit()
        logger.info("Cleared all commits from index")

    def query_by_embedding(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        after_timestamp: int | None = None,
        before_timestamp: int | None = None,
    ) -> list[dict]:
        try:
            conn = self._get_connection()

            query = "SELECT * FROM git_commits"
            conditions: list[str] = []
            params: list[int] = []

            if after_timestamp is not None:
                conditions.append("timestamp > ?")
                params.append(after_timestamp)

            if before_timestamp is not None:
                conditions.append("timestamp < ?")
                params.append(before_timestamp)

            if conditions:
                query += " WHERE " + " AND ".join(conditions)

            cursor = conn.execute(query, params)
            rows = cursor.fetchall()

            query_vec = np.array(query_embedding, dtype=np.float32)
            results = []

            for row in rows:
                embedding = self._deserialize_embedding(row["embedding"])
                score = self._cosine_similarity(query_vec, embedding)

                try:
                    files_changed = json.loads(row["files_changed"])
                except (json.JSONDecodeError, TypeError):
                    files_changed = []

                results.append({
                    "hash": row["hash"],
                    "timestamp": row["timestamp"],
                    "author": row["author"],
                    "committer": row["committer"],
                    "title": row["title"],
                    "message": row["message"],
                    "files_changed": files_changed,
                    "delta_truncated": row["delta_truncated"],
                    "score": float(score),
                    "repo_path": row["repo_path"],
                })

            results.sort(key=lambda x: x["score"], reverse=True)
            return results[:top_k]
        except sqlite3.DatabaseError as e:
            if self._is_corruption_error(e):
                self._reinitialize_after_corruption()
                return []
            raise

    def get_last_indexed_timestamp(self, repo_path: str) -> int | None:
        try:
            conn = self._get_connection()
            normalized_path = self._normalize_repo_path(repo_path)
            cursor = conn.execute(
                "SELECT MAX(timestamp) as max_ts FROM git_commits WHERE repo_path = ?",
                (normalized_path,),
            )
            row = cursor.fetchone()

            if row and row["max_ts"] is not None:
                return int(row["max_ts"])
            return None
        except sqlite3.DatabaseError as e:
            if self._is_corruption_error(e):
                self._reinitialize_after_corruption()
                return None
            raise

    def get_total_commits(self) -> int:
        try:
            conn = self._get_connection()
            cursor = conn.execute("SELECT COUNT(*) as count FROM git_commits")
            row = cursor.fetchone()
            return int(row["count"]) if row else 0
        except sqlite3.DatabaseError as e:
            if self._is_corruption_error(e):
                self._reinitialize_after_corruption()
                return 0
            raise

    @staticmethod
    def _serialize_embedding(embedding: list[float]) -> bytes:
        """Convert embedding to bytes for BLOB storage."""
        return np.array(embedding, dtype=np.float32).tobytes()

    @staticmethod
    def _deserialize_embedding(blob: bytes) -> NDArray[np.float32]:
        """Convert BLOB to numpy array."""
        return np.frombuffer(blob, dtype=np.float32)

    @staticmethod
    def _cosine_similarity(a: NDArray[np.float32], b: NDArray[np.float32]) -> float:
        """Compute cosine similarity between two vectors."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    def close(self) -> None:
        """Close database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None
            logger.debug("Closed commit indexer database connection")
