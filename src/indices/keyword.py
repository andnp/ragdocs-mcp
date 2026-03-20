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

_ARTIFACT_QUERY_RE = re.compile(r"[./\\_-]")


def _is_corruption_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(pat in msg for pat in _CORRUPTION_PATTERNS)


def _sanitize_fts_query(query: str) -> str:
    """Sanitize a query string for safe use in FTS5 MATCH."""
    # Remove FTS5 special characters (including hyphen which acts as NOT)
    sanitized = re.sub(r"[\"\'*\^(){}[\]<>|~!:\-]", " ", query)
    sanitized = sanitized.strip()
    if not sanitized:
        return '""'
    # Split into tokens and join
    tokens = sanitized.split()
    return " ".join(tokens)


def _normalize_artifact_value(value: str) -> str:
    return value.strip().lower().replace("\\", "/")


def _normalize_field_text(value: str) -> str:
    return " ".join(value.strip().lower().split())


def _has_phrase_boundary_match(text: str, phrase: str) -> bool:
    if not text or not phrase:
        return False
    return (
        text == phrase
        or text.startswith(f"{phrase} ")
        or text.endswith(f" {phrase}")
        or f" {phrase} " in text
    )


def _split_header_segments(headers: str) -> list[str]:
    return [
        normalized
        for segment in headers.split(">")
        if (normalized := _normalize_field_text(segment))
    ]


def _score_title_locality(normalized_query: str, normalized_title: str) -> float:
    if not normalized_title:
        return 0.0
    if normalized_title == normalized_query:
        return 80.0
    if _has_phrase_boundary_match(normalized_title, normalized_query):
        return 24.0
    if normalized_query in normalized_title:
        return 14.0
    return 0.0


def _score_header_locality(
    normalized_query: str,
    normalized_headers: str,
    header_segments: list[str],
) -> float:
    score = 0.0

    if normalized_headers == normalized_query:
        score = max(score, 34.0)
    elif _has_phrase_boundary_match(normalized_headers, normalized_query):
        score = max(score, 14.0)
    elif normalized_query in normalized_headers:
        score = max(score, 10.0)

    for depth, segment in enumerate(header_segments):
        depth_decay = max(0.45, 1.0 - (depth * 0.25))
        if segment == normalized_query:
            score = max(score, 44.0 * depth_decay)
            continue
        if _has_phrase_boundary_match(segment, normalized_query):
            score = max(score, 20.0 * depth_decay)
            continue
        if normalized_query in segment:
            score = max(score, 10.0 * depth_decay)

    return score


def _looks_like_artifact_query(query: str) -> bool:
    normalized = query.strip()
    if not normalized:
        return False
    return _ARTIFACT_QUERY_RE.search(normalized) is not None


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
                aliases_text = (
                    " ".join(str(a) for a in aliases_list)
                    if isinstance(aliases_list, list)
                    else str(aliases_list)
                )
                keywords_list = document.metadata.get("keywords", [])
                keywords_text = (
                    " ".join(keywords_list)
                    if isinstance(keywords_list, list)
                    else str(keywords_list)
                )
                description = str(
                    document.metadata.get("description", "")
                    or document.metadata.get("summary", "")
                )
                author = str(document.metadata.get("author", ""))
                category = str(document.metadata.get("category", ""))
                extra = " ".join(
                    filter(
                        None,
                        [aliases_text, keywords_text, description, author, category],
                    )
                )
                conn = self._conn()
                # FTS5 doesn't support real UPDATE; delete old entry first, then insert
                conn.execute(
                    "DELETE FROM search_index WHERE chunk_id = ?", (document.id,)
                )
                conn.execute(
                    """
                    INSERT INTO search_index
                        (chunk_id, doc_id, content, title, headers, tags, source_file)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        document.id,
                        document.id,
                        document.content,
                        title,
                        extra,
                        tags,
                        source_file,
                    ),
                )
                conn.commit()
            except Exception as e:
                if _is_corruption_error(e):
                    self._reinitialize_after_corruption()
                    return
                logger.warning(
                    "Failed to add document %s: %s", document.id, e, exc_info=True
                )
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
                logger.warning(
                    "Failed to add chunk %s: %s", chunk.chunk_id, e, exc_info=True
                )
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
        aliases_text = (
            " ".join(str(a) for a in aliases_list)
            if isinstance(aliases_list, list)
            else str(aliases_list)
        )
        keywords_list = metadata.get("keywords", [])
        keywords_text = (
            " ".join(keywords_list)
            if isinstance(keywords_list, list)
            else str(keywords_list)
        )
        description = str(
            metadata.get("description", "") or metadata.get("summary", "")
        )
        author = str(metadata.get("author", ""))
        category = str(metadata.get("category", ""))
        headers = " ".join(
            filter(
                None,
                [
                    header_path,
                    aliases_text,
                    keywords_text,
                    description,
                    author,
                    category,
                ],
            )
        )
        conn = self._conn()
        # FTS5 doesn't support real UPDATE; delete old entry first, then insert
        conn.execute("DELETE FROM search_index WHERE chunk_id = ?", (chunk.chunk_id,))
        conn.execute(
            """
            INSERT INTO search_index
                (chunk_id, doc_id, content, title, headers, tags, source_file)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                chunk.chunk_id,
                chunk.doc_id,
                chunk.content,
                title,
                headers,
                tags,
                source_file,
            ),
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
                logger.warning(
                    "Failed to remove document %s: %s", document_id, e, exc_info=True
                )

    def remove_chunk(self, chunk_id: str) -> None:
        """Remove a specific chunk."""
        with self._lock:
            try:
                conn = self._conn()
                conn.execute("DELETE FROM search_index WHERE chunk_id = ?", (chunk_id,))
                conn.commit()
            except Exception as e:
                if _is_corruption_error(e):
                    self._reinitialize_after_corruption()
                    return
                logger.warning(
                    "Failed to remove chunk %s: %s", chunk_id, e, exc_info=True
                )

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
                    "SELECT chunk_id FROM search_index WHERE chunk_id = ?",
                    (old_chunk_id,),
                ).fetchone()
                if row is None:
                    logger.debug("Chunk %s not found in keyword index", old_chunk_id)
                    return False

                self._insert_chunk(new_chunk)
                conn.execute(
                    "DELETE FROM search_index WHERE chunk_id = ?", (old_chunk_id,)
                )
                conn.commit()
                logger.debug(
                    "Moved chunk in keyword index: %s -> %s",
                    old_chunk_id,
                    new_chunk.chunk_id,
                )
                return True
            except Exception as e:
                if _is_corruption_error(e):
                    self._reinitialize_after_corruption()
                    return False
                logger.warning(
                    "Failed to move chunk %s: %s", old_chunk_id, e, exc_info=True
                )
                return False

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def get_chunk_by_id(self, chunk_id: str) -> dict[str, Any] | None:
        with self._lock:
            try:
                row = self._conn().execute(
                    """
                    SELECT chunk_id, doc_id, content, title, headers, tags, source_file
                    FROM search_index
                    WHERE chunk_id = ?
                    LIMIT 1
                    """,
                    (chunk_id,),
                ).fetchone()
            except sqlite3.DatabaseError as e:
                if _is_corruption_error(e):
                    self._reinitialize_after_corruption()
                    return None
                logger.warning(
                    "Chunk lookup failed for %s: %s", chunk_id, e, exc_info=True
                )
                return None

            if row is None:
                return None

            return {
                "chunk_id": row["chunk_id"],
                "doc_id": row["doc_id"],
                "content": row["content"],
                "title": row["title"],
                "headers": row["headers"],
                "tags": row["tags"],
                "source_file": row["source_file"],
            }

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
            artifact_results: list[SearchResultDict] = []
            if _looks_like_artifact_query(query):
                try:
                    artifact_results = self._search_artifact_matches(
                        query,
                        top_k * 2 if excluded_files else top_k,
                    )
                except sqlite3.DatabaseError as e:
                    if _is_corruption_error(e):
                        self._reinitialize_after_corruption()
                        return []
                    logger.warning(
                        "Artifact search lane failed for %r: %s",
                        query,
                        e,
                        exc_info=True,
                    )

            sanitized = _sanitize_fts_query(query)
            candidate_limit = max(top_k * 5, 25)

            try:
                conn = self._conn()
                rows = conn.execute(
                    """
                    SELECT chunk_id, doc_id, content, title, headers, source_file,
                           bm25(search_index) AS score
                    FROM search_index
                    WHERE search_index MATCH ?
                    ORDER BY score
                    LIMIT ?
                    """,
                    (sanitized, candidate_limit),
                ).fetchall()
            except sqlite3.DatabaseError as e:
                if _is_corruption_error(e):
                    self._reinitialize_after_corruption()
                    return []
                if "fts5" in str(e).lower() or "syntax" in str(e).lower():
                    logger.warning("FTS5 query error, falling back to LIKE: %s", e)
                    try:
                        like_query = f"%{query.strip()}%"
                        rows = (
                            self._conn()
                            .execute(
                                """
                            SELECT chunk_id, doc_id, content, title, headers,
                                   source_file, -1.0 AS score
                            FROM search_index
                            WHERE content LIKE ? OR title LIKE ? OR source_file LIKE ?
                            LIMIT ?
                            """,
                                (
                                    like_query,
                                    like_query,
                                    like_query,
                                    candidate_limit,
                                ),
                            )
                            .fetchall()
                        )
                    except Exception as e2:
                        logger.warning(
                            "LIKE fallback also failed: %s", e2, exc_info=True
                        )
                        return []
                else:
                    logger.warning("Search error: %s", e, exc_info=True)
                    return []

            ranked_rows = self._rerank_keyword_rows(query, rows)

            results: list[SearchResultDict] = []
            seen_chunk_ids: set[str] = set()

            for result in artifact_results:
                chunk_id = result["chunk_id"]
                doc_id = result["doc_id"]

                if excluded_files and docs_root:
                    from pathlib import Path as PathLib

                    from src.search.path_utils import normalize_path

                    normalized = normalize_path(doc_id, docs_root)
                    if normalized in excluded_files:
                        continue
                    if PathLib(normalized).name in excluded_files:
                        continue

                results.append(result)
                seen_chunk_ids.add(chunk_id)
                if len(results) >= top_k:
                    return results

            for row in ranked_rows:
                chunk_id = row["chunk_id"]
                doc_id = row["doc_id"]
                score = float(row["score"])

                if chunk_id in seen_chunk_ids:
                    continue

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

    def _rerank_keyword_rows(
        self,
        query: str,
        rows: list[sqlite3.Row],
    ) -> list[dict[str, Any]]:
        ranked_rows: list[dict[str, Any]] = []
        for row in rows:
            bm25_score = -float(row["score"])
            lexical_boost = self._score_field_aware_match(
                query,
                content=str(row["content"] or ""),
                title=str(row["title"] or ""),
                headers=str(row["headers"] or ""),
                source_file=str(row["source_file"] or ""),
            )
            ranked_rows.append(
                {
                    "chunk_id": row["chunk_id"],
                    "doc_id": row["doc_id"],
                    "score": bm25_score + lexical_boost,
                }
            )

        ranked_rows.sort(key=lambda item: (-float(item["score"]), str(item["chunk_id"])))
        return ranked_rows

    def _score_field_aware_match(
        self,
        query: str,
        *,
        content: str,
        title: str,
        headers: str,
        source_file: str,
    ) -> float:
        normalized_query = _normalize_field_text(query)
        if not normalized_query:
            return 0.0

        normalized_title = _normalize_field_text(title)
        normalized_headers = _normalize_field_text(headers)
        normalized_content = _normalize_field_text(content)
        normalized_source = _normalize_artifact_value(source_file)
        normalized_query_artifact = _normalize_artifact_value(query)
        basename_query = Path(normalized_query_artifact).name
        source_basename = Path(normalized_source).name if normalized_source else ""
        header_segments = _split_header_segments(headers)

        score = 0.0

        score += _score_title_locality(normalized_query, normalized_title)
        score += _score_header_locality(
            normalized_query,
            normalized_headers,
            header_segments,
        )

        if normalized_source == normalized_query_artifact:
            score += 60.0
        elif source_basename == normalized_query_artifact:
            score += 56.0
        elif basename_query and source_basename == basename_query:
            score += 52.0
        elif normalized_source.endswith(f"/{normalized_query_artifact}"):
            score += 48.0
        elif basename_query and normalized_source.endswith(f"/{basename_query}"):
            score += 44.0
        elif normalized_query_artifact in normalized_source:
            score += 16.0
        elif basename_query and basename_query in normalized_source:
            score += 12.0

        if normalized_query in normalized_content:
            score += 4.0

        return score

    def _search_artifact_matches(
        self,
        query: str,
        limit: int,
    ) -> list[SearchResultDict]:
        normalized_query = _normalize_artifact_value(query)
        basename_query = Path(normalized_query).name
        like_query = f"%{query.strip()}%"
        rows = self._conn().execute(
            """
            SELECT chunk_id, doc_id, content, title, headers, source_file
            FROM search_index
            WHERE source_file LIKE ? OR title LIKE ? OR headers LIKE ? OR content LIKE ?
            """,
            (like_query, like_query, like_query, like_query),
        ).fetchall()

        scored_rows: list[tuple[float, str, str]] = []
        for row in rows:
            content = str(row["content"] or "")
            title = str(row["title"] or "")
            headers = str(row["headers"] or "")
            source_file = str(row["source_file"] or "")
            score = self._score_artifact_match(
                normalized_query,
                basename_query,
                content,
                title,
                headers,
                source_file,
            )
            if score <= 0:
                continue
            scored_rows.append((score, row["chunk_id"], row["doc_id"]))

        scored_rows.sort(key=lambda item: (-item[0], item[1]))

        return [
            {"chunk_id": chunk_id, "doc_id": doc_id, "score": score}
            for score, chunk_id, doc_id in scored_rows[:limit]
        ]

    def _score_artifact_match(
        self,
        normalized_query: str,
        basename_query: str,
        content: str,
        title: str,
        headers: str,
        source_file: str,
    ) -> float:
        normalized_content = _normalize_artifact_value(content)
        normalized_headers = _normalize_artifact_value(headers)
        normalized_title = _normalize_artifact_value(title)
        normalized_source = _normalize_artifact_value(source_file)
        source_basename = Path(normalized_source).name if normalized_source else ""

        score = 0.0

        if normalized_source == normalized_query:
            score = max(score, 120.0)
        if source_basename == normalized_query:
            score = max(score, 115.0)
        if basename_query and source_basename == basename_query:
            score = max(score, 112.0)
        if normalized_source.endswith(f"/{normalized_query}"):
            score = max(score, 108.0)
        if basename_query and normalized_source.endswith(f"/{basename_query}"):
            score = max(score, 104.0)
        if normalized_title == normalized_query:
            score = max(score, 102.0)
        if basename_query and normalized_title == basename_query:
            score = max(score, 100.0)
        if normalized_query in normalized_source:
            score = max(score, 96.0)
        if basename_query and basename_query in normalized_source:
            score = max(score, 92.0)
        if normalized_query in normalized_title:
            score = max(score, 88.0)
        if basename_query and basename_query in normalized_title:
            score = max(score, 84.0)
        if normalized_query in normalized_headers:
            score = max(score, 82.0)
        if basename_query and basename_query in normalized_headers:
            score = max(score, 78.0)
        if normalized_query in normalized_content:
            score = max(score, 74.0)
        if basename_query and basename_query in normalized_content:
            score = max(score, 70.0)

        return score

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
                db_file,
                e,
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
                tags = (
                    ",".join(tags_list)
                    if isinstance(tags_list, list)
                    else str(tags_list)
                )
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
                logger.warning(
                    "Failed to add_document %s: %s", doc_id, e, exc_info=True
                )
                raise

    def remove_document(self, doc_id: str) -> None:
        self.remove(doc_id)

    def __len__(self) -> int:
        with self._lock:
            try:
                row = (
                    self._conn().execute("SELECT COUNT(*) FROM search_index").fetchone()
                )
                return row[0] if row else 0
            except Exception:
                return 0
