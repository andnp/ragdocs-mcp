"""Tests that `ApplicationContext._build_vector_store` reads `store.backend`.

FAISS (`VectorIndex`) must stay the default; `store.backend = "pgvector"`
must select the pgvector-backed adapter instead. This is the wiring that
makes `StoreConfig.backend` genuinely consumed by the live path, rather
than dead configuration nobody reads.
"""

from searchkernel.config import Config
from searchkernel.context import ApplicationContext
from searchkernel.indices.vector import VectorIndex


def test_defaults_to_faiss_backend():
    config = Config()

    vector = ApplicationContext._build_vector_store(config, "BAAI/bge-small-en-v1.5")

    assert isinstance(vector, VectorIndex)


def test_pgvector_backend_selects_pgvector_index(monkeypatch):
    created_kwargs = {}

    class _FakePGVectorIndex:
        def __init__(self, **kwargs):
            created_kwargs.update(kwargs)

    monkeypatch.setattr(
        "searchkernel.adapters.stores.pgvector_index.PGVectorIndex",
        _FakePGVectorIndex,
    )

    config = Config()
    config.store.backend = "pgvector"
    config.store.pg_dsn = "postgresql://example/db"

    vector = ApplicationContext._build_vector_store(config, "some-model")

    assert isinstance(vector, _FakePGVectorIndex)
    assert created_kwargs == {
        "pg_dsn": "postgresql://example/db",
        "embedding_model_name": "some-model",
    }
