"""Integration test: HuggingFace embeddings end-to-end through pgvector.

Embeds fixture DOCUMENTS with the Qwen3 provider, upserts them into the
pgvector VectorStore, then runs an ANN search with an embed_query vector
and asserts the semantically-correct document ranks #1.

Loads the real Qwen3-Embedding-0.6B model and needs a live Postgres, so it
is marked slow + serial. Set SEARCHKERNEL_PG_DSN to run it.
"""

import os
from datetime import UTC, datetime

import pytest

from searchkernel.adapters.embedding import HuggingFaceEmbeddingProvider
from searchkernel.adapters.stores.pgvector import (
    PGVectorStore,
    PostgresConnection,
    _create_schema,
)
from searchkernel.domain import Record, RecordStatus

pytestmark = [pytest.mark.slow, pytest.mark.serial]


@pytest.fixture(scope="module")
def provider() -> HuggingFaceEmbeddingProvider:
    return HuggingFaceEmbeddingProvider()


@pytest.fixture
def pg_conn():
    dsn = os.environ.get("SEARCHKERNEL_PG_DSN")
    if not dsn:
        pytest.skip("SEARCHKERNEL_PG_DSN not set")

    conn_pool = PostgresConnection(dsn)
    _create_schema(conn_pool)

    conn = conn_pool.get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT table_name FROM vector_tables;")
    for (table_name,) in cursor.fetchall():
        cursor.execute(f'DROP TABLE IF EXISTS "{table_name}";')
    cursor.execute("DELETE FROM vector_tables;")
    cursor.execute("DELETE FROM records;")
    conn.commit()
    cursor.close()
    conn_pool.put_connection(conn)

    yield conn_pool

    conn_pool.close()


def _doc(source_id: str, title: str, body: str) -> Record:
    now = datetime.now(UTC)
    return Record(
        source_kind="test",
        source_id=source_id,
        title=title,
        body=body,
        created_at=now,
        updated_at=now,
        status=RecordStatus.ACTIVE,
    )


def test_hf_embed_query_ranks_correct_doc_first(provider, pg_conn):
    assert provider.dim == 1024

    docs = [
        _doc(
            "doc:photosynthesis",
            "Photosynthesis",
            "Plants convert sunlight, water, and carbon dioxide into glucose "
            "and oxygen through photosynthesis in their chloroplasts.",
        ),
        _doc(
            "doc:database",
            "Relational Databases",
            "PostgreSQL is an open-source relational database that stores data "
            "in tables and answers queries written in SQL.",
        ),
        _doc(
            "doc:volcano",
            "Volcanoes",
            "A volcano is a rupture in the crust of a planet that allows molten "
            "rock, ash, and gases to escape from a magma chamber below.",
        ),
    ]

    doc_vectors = provider.embed([d.body for d in docs])
    for doc, vec in zip(docs, doc_vectors):
        doc.embedding = vec

    store = PGVectorStore(pg_conn)
    store.upsert(docs, model_name=provider.model_name, dim=provider.dim)

    query_vec = provider.embed_query("How do plants make food from sunlight?")
    results = store.search(
        query_vec, k=3, model_name=provider.model_name, dim=provider.dim
    )

    assert len(results) == 3
    assert results[0][0] == "doc:photosynthesis"
