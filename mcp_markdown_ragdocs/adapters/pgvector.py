"""Application-owned composition adapter for the optional pgvector backend."""

from __future__ import annotations

from typing import Any

from searchkernel.adapters.stores import PGVectorStore, PostgresConnection


def build_pgvector_index(
    *,
    pg_dsn: str,
    embedding_model_name: str,
    truncate_dim: int | None,
) -> Any:
    del embedding_model_name, truncate_dim
    return PGVectorStore(PostgresConnection(pg_dsn))
