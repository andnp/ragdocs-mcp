"""Application-owned composition adapter for the optional pgvector backend."""

from __future__ import annotations

from importlib import import_module
from typing import Any


def build_pgvector_index(
    *,
    pg_dsn: str,
    embedding_model_name: str,
    truncate_dim: int | None,
) -> Any:
    module = import_module("searchkernel.adapters.stores.pgvector_index")
    index_type = getattr(module, "PGVectorIndex")
    return index_type(
        pg_dsn=pg_dsn,
        embedding_model_name=embedding_model_name,
        truncate_dim=truncate_dim,
    )
