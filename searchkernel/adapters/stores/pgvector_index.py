"""PGVectorIndex: a VectorIndex-shaped adapter over the PGVectorStore port.

The live query/ingestion path (`IndexManager`, `SearchOrchestrator`,
`ChunkHydrator`, the pipeline stages) calls a richer surface than the
narrow `VectorStore` port exposes: `VectorStore` only models embedding
upsert/search/delete, while the live path also needs by-id chunk
hydration (content + metadata), per-document chunk enumeration, and
in-place chunk-path renames.

Rather than widen the port for one adapter (which would leak FAISS/pgvector
concerns into code that only needs `upsert`/`search`/`delete`), this class
presents the exact method surface the live path actually calls -- the same
set `searchkernel.indices.vector.VectorIndex` implements for the FAISS
backend -- and backs it with `PGVectorStore` plus direct reads of the
`records` table `PGVectorStore` already writes (chunk bookkeeping such as
doc_id/header_path/file_path/parent_chunk_id/project_id rides in
`records.metadata`, since the port itself has no by-id/by-document lookup).
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from psycopg2 import sql

from searchkernel.adapters.stores.pgvector import (
    PGVectorStore,
    PostgresConnection,
    _create_schema,
)
from searchkernel.domain import Record, RecordStatus, Vector
from searchkernel.models import Chunk
from searchkernel.search.types import SearchResultDict

logger = logging.getLogger(__name__)

# Chunks are indexed as Records under this fixed source_kind so this
# adapter's document/chunk bookkeeping queries never collide with other
# record kinds (e.g. git commits) that might share the same `records` table.
_SOURCE_KIND = "chunk"


class _Embedder(Protocol):
    model_name: str
    dim: int

    def embed(self, texts: list[str]) -> list[Vector]: ...
    def embed_query(self, text: str) -> Vector: ...


def _default_embedder(model_name: str) -> _Embedder:
    from searchkernel.adapters.embedding import HuggingFaceEmbeddingProvider

    return HuggingFaceEmbeddingProvider(model_name=model_name)


def _embedding_text(chunk: Chunk) -> str:
    return f"{chunk.header_path}\n\n{chunk.content}" if chunk.header_path else chunk.content


def _chunk_to_record(chunk: Chunk) -> Record:
    metadata = {
        **chunk.metadata,
        "doc_id": chunk.doc_id,
        "chunk_index": chunk.chunk_index,
        "header_path": chunk.header_path,
        "file_path": chunk.file_path,
        "parent_chunk_id": chunk.parent_chunk_id,
        "project_id": chunk.project_id,
    }
    return Record(
        source_kind=_SOURCE_KIND,
        source_id=chunk.chunk_id,
        title=chunk.header_path or chunk.file_path,
        body=chunk.content,
        created_at=chunk.modified_time,
        updated_at=chunk.modified_time,
        metadata=metadata,
        uri=chunk.file_path,
        status=RecordStatus.ACTIVE,
    )


def _row_to_result(record_id: str, body: str, metadata: dict, score: float) -> SearchResultDict:
    return {
        "chunk_id": record_id,
        "doc_id": metadata.get("doc_id"),
        "score": score,
        "header_path": metadata.get("header_path", ""),
        "file_path": metadata.get("file_path", ""),
        "project_id": metadata.get("project_id"),
        "content": body,
        "metadata": metadata,
    }


class PGVectorIndex:
    """Postgres + pgvector backed replacement for the live `VectorIndex`.

    Presents `add_chunk(s)`/`search`/`remove(_chunk)`/`update_chunk_path`/
    `prune_document`/`persist`/`load`/`is_ready`/`clear`/`get_chunk_by_id`/
    `get_chunk_ids_for_document`/`get_document_ids`/`get_parent_content`/
    `get_embedding_for_chunk`/`expand_query` -- the exact set the live
    query/ingestion path calls on `VectorIndex` today.
    """

    def __init__(
        self,
        pg_dsn: str,
        embedding_model_name: str = "BAAI/bge-small-en-v1.5",
        embedder: _Embedder | None = None,
    ):
        self._conn_pool = PostgresConnection(pg_dsn)
        _create_schema(self._conn_pool)
        self._store = PGVectorStore(self._conn_pool)
        self._embedder = embedder or _default_embedder(embedding_model_name)
        self._model_name = self._embedder.model_name
        self._dim = self._embedder.dim

    def warm_up(self) -> None:
        """No-op: the embedding model loads eagerly in `__init__`."""

    def add_chunk(self, chunk: Chunk) -> None:
        self.add_chunks([chunk])

    def add_chunks(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return

        vectors = self._embedder.embed([_embedding_text(c) for c in chunks])
        records = []
        for chunk, vector in zip(chunks, vectors):
            record = _chunk_to_record(chunk)
            record.embedding = vector
            records.append(record)
        self._store.upsert(records, self._model_name, self._dim)

    def search(
        self,
        query: str,
        top_k: int = 10,
        excluded_files: set[str] | None = None,
        docs_root: Path | None = None,
    ) -> list[SearchResultDict]:
        if not query.strip():
            return []

        fetch_k = top_k * 2 if excluded_files else top_k
        query_vector = self._embedder.embed_query(query)
        hits = self._store.search(
            query_vector, fetch_k, model_name=self._model_name, dim=self._dim
        )
        if not hits:
            return []

        rows = self._fetch_records([record_id for record_id, _score in hits])

        results: list[SearchResultDict] = []
        for record_id, score in hits:
            row = rows.get(record_id)
            if row is None:
                continue
            body, metadata = row

            if excluded_files and docs_root:
                file_path = metadata.get("file_path", "")
                if file_path:
                    from searchkernel.search.path_utils import matches_any_excluded

                    if matches_any_excluded(file_path, excluded_files, docs_root):
                        continue

            results.append(_row_to_result(record_id, body, metadata, score))
            if len(results) >= top_k:
                break

        return results

    def get_chunk_by_id(self, chunk_id: str) -> SearchResultDict | None:
        row = self._fetch_records([chunk_id]).get(chunk_id)
        if row is None:
            return None
        body, metadata = row
        return _row_to_result(chunk_id, body, metadata, 1.0)

    def get_parent_content(self, parent_chunk_id: str) -> str | None:
        chunk = self.get_chunk_by_id(parent_chunk_id)
        return chunk.get("content") if chunk else None

    def get_embedding_for_chunk(self, chunk_id: str) -> list[float] | None:
        chunk = self.get_chunk_by_id(chunk_id)
        if chunk is None:
            return None
        content = chunk.get("content")
        if not content:
            return None
        return self._embedder.embed([content])[0]

    def get_chunk_ids_for_document(self, doc_id: str) -> list[str]:
        conn = self._conn_pool.get_connection()
        try:
            cursor = conn.cursor()
            table_name = self._own_vector_table_name(cursor)
            if table_name is None:
                return []
            cursor.execute(
                sql.SQL(
                    "SELECT r.record_id FROM records r "
                    "JOIN {table} v ON v.record_id = r.record_id "
                    "WHERE r.source_kind = %s AND r.metadata->>'doc_id' = %s;"
                ).format(table=sql.Identifier(table_name)),
                (_SOURCE_KIND, doc_id),
            )
            return [row[0] for row in cursor.fetchall()]
        finally:
            cursor.close()
            self._conn_pool.put_connection(conn)

    def get_document_ids(self) -> list[str]:
        conn = self._conn_pool.get_connection()
        try:
            cursor = conn.cursor()
            table_name = self._own_vector_table_name(cursor)
            if table_name is None:
                return []
            cursor.execute(
                sql.SQL(
                    "SELECT DISTINCT r.metadata->>'doc_id' FROM records r "
                    "JOIN {table} v ON v.record_id = r.record_id "
                    "WHERE r.source_kind = %s;"
                ).format(table=sql.Identifier(table_name)),
                (_SOURCE_KIND,),
            )
            return [row[0] for row in cursor.fetchall() if row[0] is not None]
        finally:
            cursor.close()
            self._conn_pool.put_connection(conn)

    def _own_vector_table_name(self, cursor) -> str | None:
        """Resolve this instance's `(model_name, dim)` vector table, if any.

        Enumeration queries (`get_document_ids`/`get_chunk_ids_for_document`)
        join through this table so they only ever see chunks embedded under
        this model -- `records` is shared Postgres state, and other models
        (or unrelated corpora on the same DSN) must not leak in.
        """
        cursor.execute(
            "SELECT table_name FROM vector_tables WHERE model_name = %s AND dim = %s;",
            (self._model_name, self._dim),
        )
        row = cursor.fetchone()
        return row[0] if row else None

    def expand_query(
        self,
        query: str,
        top_k: int = 3,
        similarity_threshold: float = 0.5,
    ) -> str:
        """No-op: vocabulary-based query expansion is not implemented here.

        FAISS's `VectorIndex.expand_query` is itself a passthrough until its
        concept vocabulary has warmed up, so a cold pgvector index behaves
        identically; building the equivalent vocabulary machinery for
        pgvector is deferred (not required for the index+search cutover).
        """
        return query

    def remove(self, document_id: str) -> None:
        chunk_ids = self.get_chunk_ids_for_document(document_id)
        if chunk_ids:
            self._store.delete(chunk_ids)

    def remove_chunk(self, chunk_id: str) -> None:
        self._store.delete([chunk_id])

    def prune_document(self, doc_id: str) -> int:
        chunk_ids = self.get_chunk_ids_for_document(doc_id)
        if chunk_ids:
            self._store.delete(chunk_ids)
        return len(chunk_ids)

    def update_chunk_path(
        self, old_chunk_id: str, new_chunk_id: str, new_metadata: dict
    ) -> bool:
        row = self._fetch_records([old_chunk_id]).get(old_chunk_id)
        if row is None:
            return False
        body, _old_metadata = row

        record = Record(
            source_kind=_SOURCE_KIND,
            source_id=new_chunk_id,
            title=new_metadata.get("header_path") or new_metadata.get("file_path", ""),
            body=body,
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
            metadata=dict(new_metadata),
            uri=new_metadata.get("file_path"),
            status=RecordStatus.ACTIVE,
        )
        record.embedding = self._embedder.embed([body])[0]
        self._store.upsert([record], self._model_name, self._dim)
        self._store.delete([old_chunk_id])
        return True

    def is_ready(self) -> bool:
        return True

    def persist(self, path: Path) -> None:
        """No-op: pgvector data is already durable in Postgres."""

    def load(self, path: Path) -> None:
        """No-op: pgvector data is already durable in Postgres."""

    def clear(self) -> None:
        """Delete every record in *this* model's vector table.

        Scoped to `(model_name, dim)` rather than all `source_kind="chunk"`
        records, since multiple models can coexist in the same Postgres
        database (e.g. during a model migration) and clearing one index
        should not touch another's.
        """
        conn = self._conn_pool.get_connection()
        try:
            cursor = conn.cursor()
            table_name = self._own_vector_table_name(cursor)
            if table_name is None:
                return
            cursor.execute(
                sql.SQL("SELECT record_id FROM {table};").format(
                    table=sql.Identifier(table_name)
                )
            )
            record_ids = [r[0] for r in cursor.fetchall()]
        finally:
            cursor.close()
            self._conn_pool.put_connection(conn)
        if record_ids:
            self._store.delete(record_ids)

    def _fetch_records(self, record_ids: list[str]) -> dict[str, tuple[str, dict[str, Any]]]:
        if not record_ids:
            return {}
        conn = self._conn_pool.get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT record_id, body, metadata FROM records WHERE record_id = ANY(%s);",
                (record_ids,),
            )
            return {row[0]: (row[1], row[2] or {}) for row in cursor.fetchall()}
        finally:
            cursor.close()
            self._conn_pool.put_connection(conn)
