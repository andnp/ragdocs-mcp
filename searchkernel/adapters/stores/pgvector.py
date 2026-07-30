"""Postgres + pgvector store adapter implementing all four store ports.

Provides VectorStore (with HNSW ANN), KeywordStore (full-text search),
GraphStore (edge relationships), and CacheStore (epoch-based invalidation).
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from typing import Any

import psycopg2
import psycopg2.pool
from psycopg2 import sql

from searchkernel.domain import Record, Vector

logger = logging.getLogger(__name__)

_IDENT_RE = re.compile(r"[^a-z0-9_]+")

# Default HNSW query-time recall knob. Higher = better recall, more latency.
DEFAULT_HNSW_EF_SEARCH = 100


def _sanitize_model_name(model_name: str) -> str:
    """Turn an arbitrary model name into a safe SQL-identifier fragment.

    Appends a short content hash so distinct names that sanitize to the
    same fragment (e.g. differing only in punctuation) never collide.
    """
    lowered = model_name.lower()
    sanitized = _IDENT_RE.sub("_", lowered).strip("_") or "model"
    digest = hashlib.sha256(model_name.encode("utf-8")).hexdigest()[:8]
    return f"{sanitized}_{digest}"


def _vector_table_name(model_name: str, dim: int) -> str:
    """Deterministic per-(model_name, dim) table name.

    Each embedding model/dimension pair gets its own typed `vector(dim)`
    column and its own HNSW index, so ANN search stays index-compatible
    even as multiple models coexist (e.g. during a model migration).
    """
    return f"vectors__{_sanitize_model_name(model_name)}__{dim}"


def _vector_literal(vec: Vector) -> str:
    """Serialize a Python vector to pgvector's `[v1,v2,...]` text format."""
    return "[" + ",".join(repr(float(x)) for x in vec) + "]"


class PostgresConnection:
    """Thread-safe Postgres connection pool."""

    def __init__(self, dsn: str, min_connections: int = 2, max_connections: int = 10):
        """Initialize connection pool.

        Args:
            dsn: PostgreSQL connection string
            min_connections: Minimum idle connections in pool
            max_connections: Maximum connections in pool
        """
        self.dsn = dsn
        self.pool = psycopg2.pool.SimpleConnectionPool(
            min_connections, max_connections, dsn
        )

    def get_connection(self):
        """Get a connection from the pool."""
        return self.pool.getconn()

    def put_connection(self, conn):
        """Return a connection to the pool."""
        self.pool.putconn(conn)

    def execute(self, sql: str, params: tuple = ()) -> Any:
        """Execute a query and return results."""
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(sql, params)
            result = cursor.fetchall()
            cursor.close()
            conn.commit()
            return result
        finally:
            self.put_connection(conn)

    def execute_one(self, sql: str, params: tuple = ()) -> Any | None:
        """Execute a query and return a single result."""
        result = self.execute(sql, params)
        return result[0] if result else None

    def close(self):
        """Close all connections in the pool."""
        self.pool.closeall()


def _create_schema(conn_pool: PostgresConnection) -> None:
    """Create idempotent schema for vector, keyword, graph, and cache stores."""
    conn = conn_pool.get_connection()
    try:
        cursor = conn.cursor()
        # Create pgvector extension
        cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")

        # Registry of per-(model_name, dim) vector tables. Each embedding
        # model/dimension pair gets its own typed `vector(dim)` table with
        # a dedicated HNSW index (see PGVectorStore._ensure_vector_table),
        # so ANN search is always index-compatible even with multiple
        # models coexisting (e.g. during a model migration).
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS vector_tables (
                model_name TEXT NOT NULL,
                dim INT NOT NULL,
                table_name TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (model_name, dim)
            );
        """)

        # Records table (denormalized metadata for full-text search)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS records (
                record_id TEXT PRIMARY KEY,
                source_kind TEXT NOT NULL,
                source_id TEXT NOT NULL,
                title TEXT NOT NULL,
                body TEXT NOT NULL,
                tsvector_body tsvector,
                created_at TIMESTAMP,
                updated_at TIMESTAMP,
                metadata JSONB DEFAULT '{}',
                uri TEXT,
                status TEXT DEFAULT 'active'
            );
        """)

        # Full-text search index
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_records_tsvector
            ON records USING gin (tsvector_body);
        """)

        # Graph edges table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS graph_edges (
                source_id TEXT NOT NULL,
                target_id TEXT NOT NULL,
                edge_type TEXT NOT NULL,
                weight REAL DEFAULT 1.0,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (source_id, target_id, edge_type)
            );
        """)

        # Graph edges index
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_graph_edges_source
            ON graph_edges (source_id);
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_graph_edges_target
            ON graph_edges (target_id);
        """)

        # Cache store table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS cache_store (
                key TEXT PRIMARY KEY,
                value JSONB NOT NULL,
                epoch INT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)

        # Cache epoch index
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_cache_epoch
            ON cache_store (epoch);
        """)

        # Index epoch tracking
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS index_epoch (
                id INT PRIMARY KEY DEFAULT 1,
                epoch INT DEFAULT 0,
                CONSTRAINT only_one_row CHECK (id = 1)
            );
        """)

        # Initialize epoch if not present
        cursor.execute("SELECT COUNT(*) FROM index_epoch;")
        if cursor.fetchone()[0] == 0:
            cursor.execute("INSERT INTO index_epoch (epoch) VALUES (0);")

        conn.commit()
        logger.info("pgvector schema initialized successfully")
    finally:
        cursor.close()
        conn_pool.put_connection(conn)


class PGVectorStore:
    """Postgres + pgvector implementation of VectorStore port.

    Each (model_name, dim) pair gets its own table with a typed
    `vector(dim)` column and a dedicated HNSW index, so ANN search is
    always index-compatible -- pgvector's HNSW requires a fixed
    dimension per indexed column, which an untyped `vector` column
    cannot provide. Multiple models can coexist (e.g. during a model
    migration); each lives in its own table.

    `search()` takes `model_name`/`dim` explicitly (rather than relying on
    instance "active model" state) so concurrent callers can query
    different models safely.
    """

    def __init__(
        self,
        conn_pool: PostgresConnection,
        hnsw_ef_search: int = DEFAULT_HNSW_EF_SEARCH,
    ):
        """Initialize vector store.

        Args:
            conn_pool: PostgresConnection pool
            hnsw_ef_search: hnsw.ef_search GUC applied per query (recall/latency knob)
        """
        self.conn_pool = conn_pool
        self.hnsw_ef_search = hnsw_ef_search

    def _ensure_vector_table(self, cursor, model_name: str, dim: int) -> str:
        """Return the table name for (model_name, dim), creating it (+ HNSW index) if needed.

        Raises:
            ValueError: If model_name is already registered under a different dim.
        """
        cursor.execute(
            "SELECT dim, table_name FROM vector_tables WHERE model_name = %s;",
            (model_name,),
        )
        rows = cursor.fetchall()
        for existing_dim, existing_table in rows:
            if existing_dim != dim:
                raise ValueError(
                    f"Dimension mismatch for model {model_name}: "
                    f"expected {existing_dim}, got {dim}"
                )
            return existing_table

        table_name = _vector_table_name(model_name, dim)

        cursor.execute(
            sql.SQL(
                "CREATE TABLE IF NOT EXISTS {table} ("
                "record_id TEXT PRIMARY KEY, "
                "embedding vector({dim}) NOT NULL, "
                "created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, "
                "updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP"
                ");"
            ).format(table=sql.Identifier(table_name), dim=sql.SQL(str(int(dim))))
        )

        cursor.execute(
            sql.SQL(
                "CREATE INDEX IF NOT EXISTS {index_name} ON {table} "
                "USING hnsw (embedding vector_cosine_ops);"
            ).format(
                index_name=sql.Identifier(f"idx_{table_name}_hnsw"),
                table=sql.Identifier(table_name),
            )
        )

        cursor.execute(
            "INSERT INTO vector_tables (model_name, dim, table_name) "
            "VALUES (%s, %s, %s) ON CONFLICT (model_name, dim) DO NOTHING;",
            (model_name, dim, table_name),
        )
        return table_name

    def upsert(self, records: list[Record], model_name: str, dim: int) -> None:
        """Upsert records with embeddings.

        Args:
            records: Records with embedding set
            model_name: Embedding model name
            dim: Vector dimensionality

        Raises:
            ValueError: If embedding dimension doesn't match stored dimension
        """
        if not records:
            return

        conn = self.conn_pool.get_connection()
        try:
            cursor = conn.cursor()

            table_name = self._ensure_vector_table(cursor, model_name, dim)

            # Upsert records table
            for record in records:
                metadata_json = json.dumps(record.metadata)
                tsvector_text = f"{record.title} {record.body}"

                cursor.execute(
                    """
                    INSERT INTO records
                    (record_id, source_kind, source_id, title, body,
                     tsvector_body, created_at, updated_at, metadata, uri, status)
                    VALUES (%s, %s, %s, %s, %s,
                            to_tsvector('english', %s), %s, %s, %s, %s, %s)
                    ON CONFLICT (record_id) DO UPDATE SET
                        title = EXCLUDED.title,
                        body = EXCLUDED.body,
                        tsvector_body = to_tsvector('english', EXCLUDED.title || ' ' || EXCLUDED.body),
                        updated_at = EXCLUDED.updated_at,
                        metadata = EXCLUDED.metadata;
                    """,
                    (
                        record.source_id,
                        record.source_kind,
                        record.source_id,
                        record.title,
                        record.body,
                        tsvector_text,
                        record.created_at,
                        record.updated_at,
                        metadata_json,
                        record.uri,
                        record.status.value,
                    ),
                )

            # Upsert vectors into the per-model typed table
            upsert_vec_sql = sql.SQL(
                "INSERT INTO {table} (record_id, embedding) "
                "VALUES (%s, %s::vector) "
                "ON CONFLICT (record_id) DO UPDATE SET "
                "embedding = EXCLUDED.embedding, updated_at = CURRENT_TIMESTAMP;"
            ).format(table=sql.Identifier(table_name))

            for record in records:
                if record.embedding is None:
                    continue
                if len(record.embedding) != dim:
                    raise ValueError(
                        f"Embedding dimension mismatch for record {record.source_id}: "
                        f"expected {dim}, got {len(record.embedding)}"
                    )
                cursor.execute(
                    upsert_vec_sql,
                    (record.source_id, _vector_literal(record.embedding)),
                )

            # Increment epoch
            cursor.execute("UPDATE index_epoch SET epoch = epoch + 1;")

            conn.commit()
            logger.debug(f"Upserted {len(records)} records for model {model_name}")
        finally:
            cursor.close()
            self.conn_pool.put_connection(conn)

    def search(
        self,
        query_vector: Vector,
        k: int,
        *,
        model_name: str,
        dim: int,
        filters: dict[str, Any] | None = None,
    ) -> list[tuple[str, float]]:
        """Search for nearest neighbors using cosine similarity (ANN via HNSW).

        Args:
            query_vector: Query embedding vector
            k: Number of results to return
            model_name: Embedding model query_vector was produced with;
                selects which per-model table to search.
            dim: Dimensionality of query_vector.
            filters: Optional filters (source-kind filtering, etc.)

        Returns:
            List of (record_id, similarity_score) tuples, sorted descending
        """
        conn = self.conn_pool.get_connection()
        try:
            cursor = conn.cursor()

            cursor.execute(
                "SELECT 1 FROM vector_tables WHERE model_name = %s AND dim = %s;",
                (model_name, dim),
            )
            if cursor.fetchone() is None:
                return []

            table_name = _vector_table_name(model_name, dim)

            # hnsw.ef_search cannot be a bind parameter (SET does not accept
            # protocol parameters); the value is an internally-controlled int.
            cursor.execute(f"SET LOCAL hnsw.ef_search = {int(self.hnsw_ef_search)};")

            where_clause = ""
            kind_params: list[Any] = []
            if filters and "source_kinds" in filters:
                source_kinds = filters["source_kinds"]
                placeholders = ",".join(["%s"] * len(source_kinds))
                where_clause = f"AND r.source_kind IN ({placeholders})"
                kind_params = list(source_kinds)

            vec_literal = _vector_literal(query_vector)

            # Order by the raw distance operator (not a wrapped/aliased
            # expression) so the planner can use the HNSW index for ANN.
            query_sql = sql.SQL(
                "SELECT v.record_id, v.embedding <=> %s::vector AS distance "
                "FROM {table} v "
                "JOIN records r ON v.record_id = r.record_id "
                "WHERE 1 = 1 " + where_clause + " "
                "ORDER BY v.embedding <=> %s::vector ASC "
                "LIMIT %s;"
            ).format(table=sql.Identifier(table_name))

            params = [vec_literal, *kind_params, vec_literal, k]

            cursor.execute(query_sql, params)
            results = cursor.fetchall()
            conn.commit()
            return [(row[0], 1.0 - float(row[1])) for row in results]
        finally:
            cursor.close()
            self.conn_pool.put_connection(conn)

    def delete(self, record_ids: list[str]) -> None:
        """Delete records by ID.

        Args:
            record_ids: IDs to delete
        """
        if not record_ids:
            return

        conn = self.conn_pool.get_connection()
        try:
            cursor = conn.cursor()

            cursor.execute("SELECT DISTINCT table_name FROM vector_tables;")
            table_names = [row[0] for row in cursor.fetchall()]

            for table_name in table_names:
                cursor.execute(
                    sql.SQL("DELETE FROM {table} WHERE record_id = ANY(%s);").format(
                        table=sql.Identifier(table_name)
                    ),
                    (record_ids,),
                )

            cursor.execute(
                "DELETE FROM records WHERE record_id = ANY(%s);", (record_ids,)
            )

            # Increment epoch
            cursor.execute("UPDATE index_epoch SET epoch = epoch + 1;")

            conn.commit()
            logger.debug(f"Deleted {len(record_ids)} records")
        finally:
            cursor.close()
            self.conn_pool.put_connection(conn)

    def epoch(self) -> int:
        """Get current index epoch."""
        conn = self.conn_pool.get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT epoch FROM index_epoch LIMIT 1;")
            result = cursor.fetchone()
            return result[0] if result else 0
        finally:
            cursor.close()
            self.conn_pool.put_connection(conn)


class PGKeywordStore:
    """Postgres full-text search implementation of KeywordStore port."""

    def __init__(self, conn_pool: PostgresConnection):
        """Initialize keyword store.

        Args:
            conn_pool: PostgresConnection pool
        """
        self.conn_pool = conn_pool

    def index(self, records: list[Record]) -> None:
        """Index records for full-text search.

        Args:
            records: Records to index
        """
        if not records:
            return

        conn = self.conn_pool.get_connection()
        try:
            cursor = conn.cursor()

            for record in records:
                tsvector_text = f"{record.title} {record.body}"
                cursor.execute(
                    """
                    UPDATE records
                    SET tsvector_body = to_tsvector('english', %s)
                    WHERE record_id = %s;
                    """,
                    (tsvector_text, record.source_id),
                )

            conn.commit()
            logger.debug(f"Indexed {len(records)} records for keyword search")
        finally:
            cursor.close()
            self.conn_pool.put_connection(conn)

    def search(
        self, query: str, k: int, filters: dict[str, Any] | None = None
    ) -> list[tuple[str, float]]:
        """Search for records matching the query.

        Args:
            query: Full-text search query
            k: Number of results to return
            filters: Optional filters

        Returns:
            List of (record_id, relevance_score) tuples, sorted descending
        """
        conn = self.conn_pool.get_connection()
        try:
            cursor = conn.cursor()

            # Build WHERE clause for filters
            where_clause = ""
            params = [query]
            if filters and "source_kinds" in filters:
                source_kinds = filters["source_kinds"]
                placeholders = ",".join(["%s"] * len(source_kinds))
                where_clause = f"AND source_kind IN ({placeholders})"
                params.extend(source_kinds)

            sql = f"""
                SELECT record_id,
                       ts_rank(tsvector_body, plainto_tsquery('english', %s)) as relevance
                FROM records
                WHERE tsvector_body @@ plainto_tsquery('english', %s)
                {where_clause}
                ORDER BY relevance DESC
                LIMIT %s;
            """
            params.insert(1, query)
            params.append(k)

            cursor.execute(sql, params)
            results = cursor.fetchall()
            return [(row[0], float(row[1])) for row in results]
        finally:
            cursor.close()
            self.conn_pool.put_connection(conn)


class PGGraphStore:
    """Postgres implementation of GraphStore port."""

    def __init__(self, conn_pool: PostgresConnection):
        """Initialize graph store.

        Args:
            conn_pool: PostgresConnection pool
        """
        self.conn_pool = conn_pool

    def upsert_edges(
        self,
        edges: list[tuple[str, str, str, float]],
    ) -> None:
        """Upsert edges in the graph.

        Args:
            edges: List of (source_id, target_id, edge_type, weight) tuples
        """
        if not edges:
            return

        conn = self.conn_pool.get_connection()
        try:
            cursor = conn.cursor()

            for source_id, target_id, edge_type, weight in edges:
                cursor.execute(
                    """
                    INSERT INTO graph_edges (source_id, target_id, edge_type, weight)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (source_id, target_id, edge_type) DO UPDATE SET
                        weight = EXCLUDED.weight,
                        updated_at = CURRENT_TIMESTAMP;
                    """,
                    (source_id, target_id, edge_type, weight),
                )

            conn.commit()
            logger.debug(f"Upserted {len(edges)} edges")
        finally:
            cursor.close()
            self.conn_pool.put_connection(conn)

    def neighbors(
        self,
        record_id: str,
        edge_types: list[str] | None = None,
        depth: int = 1,
    ) -> list[tuple[str, str, float]]:
        """Retrieve neighbors of a record.

        Args:
            record_id: Starting record ID
            edge_types: Optional filter by edge type
            depth: Traversal depth (default 1 for one-hop)

        Returns:
            List of (neighbor_id, edge_type, cumulative_weight) tuples
        """
        conn = self.conn_pool.get_connection()
        try:
            cursor = conn.cursor()

            # For now, just do one-hop neighbors (depth=1)
            # A full recursive solution would use CTEs for deeper traversals
            where_clause = ""
            params = [record_id]

            if edge_types:
                placeholders = ",".join(["%s"] * len(edge_types))
                where_clause = f"AND edge_type IN ({placeholders})"
                params.extend(edge_types)

            sql = f"""
                SELECT target_id, edge_type, weight
                FROM graph_edges
                WHERE source_id = %s {where_clause}
                ORDER BY weight DESC;
            """

            cursor.execute(sql, params)
            results = cursor.fetchall()
            return [(row[0], row[1], float(row[2])) for row in results]
        finally:
            cursor.close()
            self.conn_pool.put_connection(conn)


class PGCacheStore:
    """Postgres implementation of CacheStore port with epoch-based invalidation."""

    def __init__(self, conn_pool: PostgresConnection):
        """Initialize cache store.

        Args:
            conn_pool: PostgresConnection pool
        """
        self.conn_pool = conn_pool

    def get(self, key: str) -> Any | None:
        """Retrieve a cached value.

        Args:
            key: Cache key

        Returns:
            Cached value, or None if not found or stale
        """
        conn = self.conn_pool.get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT value FROM cache_store WHERE key = %s;", (key,))
            result = cursor.fetchone()
            if result:
                value = result[0]
                # psycopg2 automatically deserializes JSONB columns to dicts
                if isinstance(value, str):
                    return json.loads(value)
                return value
            return None
        finally:
            cursor.close()
            self.conn_pool.put_connection(conn)

    def set(self, key: str, value: Any, epoch: int) -> None:
        """Store a value with an associated epoch.

        Args:
            key: Cache key
            value: Value to cache
            epoch: Index epoch at cache time
        """
        conn = self.conn_pool.get_connection()
        try:
            cursor = conn.cursor()
            value_json = json.dumps(value)
            cursor.execute(
                """
                INSERT INTO cache_store (key, value, epoch)
                VALUES (%s, %s, %s)
                ON CONFLICT (key) DO UPDATE SET
                    value = EXCLUDED.value,
                    epoch = EXCLUDED.epoch;
                """,
                (key, value_json, epoch),
            )
            conn.commit()
        finally:
            cursor.close()
            self.conn_pool.put_connection(conn)

    def invalidate_epoch(self, epoch: int) -> None:
        """Invalidate all entries from an epoch or earlier.

        Args:
            epoch: Entries with epoch <= this are discarded
        """
        conn = self.conn_pool.get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(
                "DELETE FROM cache_store WHERE epoch <= %s;",
                (epoch,),
            )
            conn.commit()
            logger.debug(f"Invalidated cache entries for epochs <= {epoch}")
        finally:
            cursor.close()
            self.conn_pool.put_connection(conn)
