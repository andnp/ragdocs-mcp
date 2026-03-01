from __future__ import annotations

import logging
import re
import sqlite3
import threading
from pathlib import Path
from typing import Any

from src.models import Chunk, Document
from src.search.types import SearchResultDict
from src.storage.db import DatabaseManager

logger = logging.getLogger(__name__)

_CORRUPTION_PATTERNS = (
    "database disk image is malformed",
    "disk i/o error",
    "file is not a database",
    "database is locked",
    "unable to open database file",
)


def _is_corruption_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(pat in msg for pat in _CORRUPTION_PATTERNS)


def _sanitize_fts_query(query: str) -> str:
    """Sanitize a query string for safe use in FTS5 MATCH."""
    # Remove FTS5 special characters (including hyphen which acts as NOT)
    sanitized = re.sub(r'[\"\'*\^(){}[\]<>|~!:\-]', ' ', query)
    sanitized = sanitized.strip()
    if not sanitized:
        return '""'
    # Split into tokens and join
    tokens = sanitized.split()
    return " ".join(tokens)


class KeywordIndex:
    def __init__(self, db_manager: DatabaseManager | None = None) -> None:
        if db_manager is None:
            import tempfile
            _tmp = tempfile.mkdtemp(prefix="keyword_idx_")
            db_manager = DatabaseManager(Path(_tmp) / "index.db")
        self._db = db_manager
        self._lock = threading.Lock()

    def _conn(self) -> sqlite3.Connection:
        return self._db.get_connection()

    def _reinitialize_after_corruption(self) -> None:
        """Delete the corrupt DB file and reset to a fresh in-memory index.

        Reconciliation will repopulate the index from source documents.
        """
        corrupt_path = self._db._db_path
        logger.warning(
            "Keyword index corruption detected at %s; reinitializing clean "
            "(reconciliation will repopulate from source documents).",
            corrupt_path,
        )
        try:
            self._db.close()
        except Exception:
            pass
        for suffix in ("", "-wal", "-shm"):
            p = corrupt_path.with_suffix(".db" + suffix) if suffix else corrupt_path
            try:
                if p.exists():
                    p.unlink()
            except OSError as e:
                logger.warning("Could not delete %s: %s", p, e)
        import tempfile
        _tmp = tempfile.mkdtemp(prefix="keyword_idx_")
        self._db = DatabaseManager(Path(_tmp) / "index.db")

    # ------------------------------------------------------------------
    # Write methods
    # ------------------------------------------------------------------

    def add(self, document: Document) -> None:
        """Add a document as a single FTS5 entry."""
        with self._lock:
            try:
                title = str(document.metadata.get("title", ""))
                tags_list = document.tags if document.tags else []
                tags = ",".join(tags_list)
                source_file = str(document.metadata.get("source_file", document.id))
                # Include aliases, keywords, description, author, category in indexed headers field
                aliases_list = document.metadata.get("aliases", [])
                aliases_text = " ".join(str(a) for a in aliases_list) if isinstance(aliases_list, list) else str(aliases_list)
                keywords_list = document.metadata.get("keywords", [])
                keywords_text = " ".join(keywords_list) if isinstance(keywords_list, list) else str(keywords_list)
                description = str(document.metadata.get("description", "") or document.metadata.get("summary", ""))
                author = str(document.metadata.get("author", ""))
                category = str(document.metadata.get("category", ""))
                extra = " ".join(filter(None, [aliases_text, keywords_text, description, author, category]))
                conn = self._conn()
                # FTS5 doesn't support real UPDATE; delete old entry first, then insert
                conn.execute("DELETE FROM search_index WHERE chunk_id = ?", (document.id,))
                conn.execute(
                    """
                    INSERT INTO search_index
                        (chunk_id, doc_id, content, title, headers, tags, source_file)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (document.id, document.id, document.content, title, extra, tags, source_file),
                )
                conn.commit()
            except Exception as e:
                if _is_corruption_error(e):
                    self._reinitialize_after_corruption()
                    return
                logger.warning("Failed to add document %s: %s", document.id, e, exc_info=True)
                raise

    def add_chunk(self, chunk: Chunk) -> None:
        """Add a single chunk to FTS5 index."""
        with self._lock:
            try:
                self._insert_chunk(chunk)
                self._conn().commit()
            except Exception as e:
                if _is_corruption_error(e):
                    self._reinitialize_after_corruption()
                    return
                logger.warning("Failed to add chunk %s: %s", chunk.chunk_id, e, exc_info=True)
                raise

    def add_chunks(self, chunks: list[Chunk]) -> None:
        """Add multiple chunks in a single transaction."""
        if not chunks:
            return
        with self._lock:
            try:
                for chunk in chunks:
                    self._insert_chunk(chunk)
                self._conn().commit()
            except Exception as e:
                if _is_corruption_error(e):
                    self._reinitialize_after_corruption()
                    return
                logger.warning("Failed to add chunks: %s", e, exc_info=True)
                raise

    def _insert_chunk(self, chunk: Chunk) -> None:
        metadata = chunk.metadata
        title = str(metadata.get("title", ""))
        header_path = chunk.header_path or ""
        tags_list = metadata.get("tags", [])
        tags = ",".join(tags_list) if isinstance(tags_list, list) else str(tags_list)
        source_file = str(metadata.get("source_file", chunk.doc_id))
        # Include aliases, keywords, description, author, category in headers field
        aliases_list = metadata.get("aliases", [])
        aliases_text = " ".join(str(a) for a in aliases_list) if isinstance(aliases_list, list) else str(aliases_list)
        keywords_list = metadata.get("keywords", [])
        keywords_text = " ".join(keywords_list) if isinstance(keywords_list, list) else str(keywords_list)
        description = str(metadata.get("description", "") or metadata.get("summary", ""))
        author = str(metadata.get("author", ""))
        category = str(metadata.get("category", ""))
        headers = " ".join(filter(None, [header_path, aliases_text, keywords_text, description, author, category]))
        conn = self._conn()
        # FTS5 doesn't support real UPDATE; delete old entry first, then insert
        conn.execute("DELETE FROM search_index WHERE chunk_id = ?", (chunk.chunk_id,))
        conn.execute(
            """
            INSERT INTO search_index
                (chunk_id, doc_id, content, title, headers, tags, source_file)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (chunk.chunk_id, chunk.doc_id, chunk.content, title, headers, tags, source_file),
        )

    def remove(self, document_id: str) -> None:
        """Remove all chunks for a document."""
        with self._lock:
            try:
                conn = self._conn()
                conn.execute(
                    "DELETE FROM search_index WHERE doc_id = ?", (document_id,)
                )
                # Also delete if chunk_id == document_id (document-level entries)
                conn.execute(
                    "DELETE FROM search_index WHERE chunk_id = ?", (document_id,)
                )
                conn.commit()
            except Exception as e:
                if _is_corruption_error(e):
                    self._reinitialize_after_corruption()
                    return
                logger.warning("Failed to remove document %s: %s", document_id, e, exc_info=True)

    def remove_chunk(self, chunk_id: str) -> None:
        """Remove a specific chunk."""
        with self._lock:
            try:
                conn = self._conn()
                conn.execute(
                    "DELETE FROM search_index WHERE chunk_id = ?", (chunk_id,)
                )
                conn.commit()
            except Exception as e:
                if _is_corruption_error(e):
                    self._reinitialize_after_corruption()
                    return
                logger.warning("Failed to remove chunk %s: %s", chunk_id, e, exc_info=True)

    def remove_chunks(self, chunk_ids: list[str]) -> None:
        """Remove multiple chunks."""
        if not chunk_ids:
            return
        with self._lock:
            try:
                conn = self._conn()
                placeholders = ",".join("?" * len(chunk_ids))
                conn.execute(
                    f"DELETE FROM search_index WHERE chunk_id IN ({placeholders})",
                    chunk_ids,
                )
                conn.commit()
            except Exception as e:
                if _is_corruption_error(e):
                    self._reinitialize_after_corruption()
                    return
                logger.warning("Failed to remove chunks: %s", e, exc_info=True)

    def move_chunk(self, old_chunk_id: str, new_chunk: Chunk) -> bool:
        """Move chunk to new ID with updated metadata."""
        with self._lock:
            try:
                conn = self._conn()
                row = conn.execute(
                    "SELECT chunk_id FROM search_index WHERE chunk_id = ?", (old_chunk_id,)
                ).fetchone()
                if row is None:
                    logger.debug("Chunk %s not found in keyword index", old_chunk_id)
                    return False

                self._insert_chunk(new_chunk)
                conn.execute(
                    "DELETE FROM search_index WHERE chunk_id = ?", (old_chunk_id,)
                )
                conn.commit()
                logger.debug("Moved chunk in keyword index: %s -> %s", old_chunk_id, new_chunk.chunk_id)
                return True
            except Exception as e:
                if _is_corruption_error(e):
                    self._reinitialize_after_corruption()
                    return False
                logger.warning("Failed to move chunk %s: %s", old_chunk_id, e, exc_info=True)
                return False

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        top_k: int = 10,
        excluded_files: set[str] | None = None,
        docs_root: Path | None = None,
    ) -> list[SearchResultDict]:
        if not query.strip():
            return []

        with self._lock:
            sanitized = _sanitize_fts_query(query)

            try:
                conn = self._conn()
                rows = conn.execute(
                    """
                    SELECT chunk_id, doc_id, bm25(search_index) AS score
                    FROM search_index
                    WHERE search_index MATCH ?
                    ORDER BY score
                    LIMIT ?
                    """,
                    (sanitized, top_k * 2 if excluded_files else top_k),
                ).fetchall()
            except sqlite3.DatabaseError as e:
                if _is_corruption_error(e):
                    self._reinitialize_after_corruption()
                    return []
                if "fts5" in str(e).lower() or "syntax" in str(e).lower():
                    logger.warning("FTS5 query error, falling back to LIKE: %s", e)
                    try:
                        like_query = f"%{query.strip()}%"
                        rows = self._conn().execute(
                            """
                            SELECT chunk_id, doc_id, -1.0 AS score
                            FROM search_index
                            WHERE content LIKE ? OR title LIKE ?
                            LIMIT ?
                            """,
                            (like_query, like_query, top_k * 2 if excluded_files else top_k),
                        ).fetchall()
                    except Exception as e2:
                        logger.warning("LIKE fallback also failed: %s", e2, exc_info=True)
                        return []
                else:
                    logger.warning("Search error: %s", e, exc_info=True)
                    return []

            results: list[SearchResultDict] = []
            for row in rows:
                chunk_id = row["chunk_id"]
                doc_id = row["doc_id"]
                # bm25() returns negative values; negate so higher = better
                score = -float(row["score"])

                if excluded_files and docs_root:
                    from src.search.path_utils import normalize_path
                    from pathlib import Path as PathLib

                    normalized = normalize_path(doc_id, docs_root)
                    if normalized in excluded_files:
                        continue
                    if PathLib(normalized).name in excluded_files:
                        continue

                results.append({"chunk_id": chunk_id, "doc_id": doc_id, "score": score})
                if len(results) >= top_k:
                    break

            return results

    # ------------------------------------------------------------------
    # Persistence no-ops (data lives in SQLite)
    # ------------------------------------------------------------------

    def persist(self, path: Path) -> None:
        """Copy SQLite index to the given path directory."""
        import shutil
        path.mkdir(parents=True, exist_ok=True)
        src = self._db._db_path
        dest = path / "index.db"
        if src != dest and src.exists():
            with self._lock:
                # WAL checkpoint before copy for consistency
                try:
                    self._conn().execute("PRAGMA wal_checkpoint(FULL)")
                except Exception:
                    pass
                shutil.copy2(src, dest)

    def persist_to(self, snapshot_dir: Path) -> None:
        self.persist(snapshot_dir)

    def load_from(self, snapshot_dir: Path) -> bool:
        """Load SQLite index from snapshot directory.

        Returns False if the directory is empty, contains only Whoosh files,
        or doesn't have a valid index.db.
        Returns True if a valid index.db was found and loaded.
        """
        if not snapshot_dir.exists():
            return True  # No snapshot yet; fresh start is fine
        db_file = snapshot_dir / "index.db"
        if db_file.exists():
            self._load_from_db_file(db_file)
            return True
        # Directory exists but no index.db - could be old Whoosh snapshot or empty
        return False

    def _load_from_db_file(self, db_file: Path) -> None:
        """Reinitialize db_manager to use the given SQLite file, or reinitialize fresh on corruption."""
        candidate: DatabaseManager | None = None
        try:
            candidate = DatabaseManager(db_file)
            result = candidate.get_connection().execute("PRAGMA quick_check").fetchone()
            if result is None or result[0] != "ok":
                raise sqlite3.DatabaseError(f"quick_check returned: {result}")
        except (sqlite3.DatabaseError, sqlite3.OperationalError) as e:
            logger.warning(
                "Keyword index %s is corrupted (%s); reinitializing clean.",
                db_file, e,
                exc_info=True,
            )
            if candidate is not None:
                try:
                    candidate.close()
                except Exception:
                    pass
            db_file.unlink(missing_ok=True)
            return  # self._db remains the existing clean temp DB
        self._db.close()
        self._db = candidate

    def load(self, path: Path) -> None:
        """Load index from path directory if it contains an index.db."""
        if not path.exists():
            return
        db_file = path / "index.db"
        if db_file.exists():
            self._load_from_db_file(db_file)

    def save(self, path: Path) -> None:
        self.persist(path)

    # ------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------

    def clear(self) -> None:
        with self._lock:
            try:
                conn = self._conn()
                conn.execute("DELETE FROM search_index")
                conn.commit()
            except Exception as e:
                logger.warning("Failed to clear search_index: %s", e, exc_info=True)

    def add_document(self, doc_id: str, content: str, metadata: dict[str, Any]) -> None:
        """Add a document by id/content/metadata (IndexProtocol compatibility)."""
        with self._lock:
            try:
                title = str(metadata.get("title", ""))
                tags_list = metadata.get("tags", [])
                tags = ",".join(tags_list) if isinstance(tags_list, list) else str(tags_list)
                source_file = str(metadata.get("source_file", doc_id))
                conn = self._conn()
                conn.execute("DELETE FROM search_index WHERE chunk_id = ?", (doc_id,))
                conn.execute(
                    """
                    INSERT INTO search_index
                        (chunk_id, doc_id, content, title, headers, tags, source_file)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (doc_id, doc_id, content, title, "", tags, source_file),
                )
                conn.commit()
            except Exception as e:
                logger.warning("Failed to add_document %s: %s", doc_id, e, exc_info=True)
                raise

    def remove_document(self, doc_id: str) -> None:
        self.remove(doc_id)

    def __len__(self) -> int:
        with self._lock:
            try:
                row = self._conn().execute(
                    "SELECT COUNT(*) FROM search_index"
                ).fetchone()
                return row[0] if row else 0
            except Exception:
                return 0
