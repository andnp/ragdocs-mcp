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
    
    def _get_connection(self) -> sqlite3.Connection:
        """Get or create database connection."""
        if self._conn is None:
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
            # Enable WAL mode for better concurrency
            self._conn.execute("PRAGMA journal_mode=WAL")
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
        """
        Add or update commit in index.
        
        Args:
            hash: Commit SHA
            timestamp: Unix timestamp
            author: Author string
            committer: Committer string
            title: First line of commit message
            message: Full commit message body
            files_changed: List of changed file paths
            delta_truncated: Truncated diff text
            commit_document: Full searchable text for embedding
            repo_path: Path to repository (optional)
        """
        # Generate embedding
        embedding = self._embedding_model.get_text_embedding(commit_document)
        embedding_bytes = self._serialize_embedding(embedding)
        
        # Store in SQLite
        conn = self._get_connection()
        indexed_at = int(time.time())
        
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
                repo_path,
            ),
        )
        conn.commit()
        logger.debug(f"Indexed commit {hash[:8]}")
    
    def remove_commit(self, commit_hash: str) -> None:
        """Remove commit from index by hash."""
        conn = self._get_connection()
        conn.execute("DELETE FROM git_commits WHERE hash = ?", (commit_hash,))
        conn.commit()
        logger.debug(f"Removed commit {commit_hash[:8]}")
    
    def query_by_embedding(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        after_timestamp: int | None = None,
        before_timestamp: int | None = None,
    ) -> list[dict]:
        """
        Query commits by embedding similarity.
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            after_timestamp: Optional filter for commits after timestamp
            before_timestamp: Optional filter for commits before timestamp
        
        Returns:
            List of dicts with keys: hash, score, timestamp, etc.
        """
        conn = self._get_connection()
        
        # Build query with optional timestamp filters
        query = "SELECT * FROM git_commits"
        conditions = []
        params = []
        
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
        
        # Compute cosine similarity for each commit
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
        
        # Sort by score descending and take top K
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]
    
    def get_last_indexed_timestamp(self, repo_path: str) -> int | None:
        """Get most recent commit timestamp for a repository."""
        conn = self._get_connection()
        cursor = conn.execute(
            "SELECT MAX(timestamp) as max_ts FROM git_commits WHERE repo_path = ?",
            (repo_path,),
        )
        row = cursor.fetchone()
        
        if row and row["max_ts"] is not None:
            return int(row["max_ts"])
        return None
    
    def get_total_commits(self) -> int:
        """Count total commits in index."""
        conn = self._get_connection()
        cursor = conn.execute("SELECT COUNT(*) as count FROM git_commits")
        row = cursor.fetchone()
        return int(row["count"]) if row else 0
    
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
