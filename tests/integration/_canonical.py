"""Canonical record-kernel helpers shared by integration tests."""

from __future__ import annotations

import os
from datetime import UTC, datetime
from pathlib import Path

from searchkernel.api import RecordIdentity, build_local_record_kernel
from searchkernel.domain import Record, RecordStatus
from searchkernel.embeddings import TEST_FAKE_EMBEDDINGS_ENV_VAR

from mcp_markdown_ragdocs.config import Config
from mcp_markdown_ragdocs.app.search import build_record_search_policy
from mcp_markdown_ragdocs.indexing.record_manager import (
    RecordIndexManager,
    build_embedding_provider,
    install_bidirectional_graph_store,
)


def make_record_index_manager(
    config: Config,
    *,
    documents_roots: list[Path] | None = None,
) -> RecordIndexManager:
    """Build the same local record stack used by the application context."""

    os.environ.setdefault(TEST_FAKE_EMBEDDINGS_ENV_VAR, "1")
    model_name = config.embedding.model_name
    embedding_provider = build_embedding_provider(config, model_name)
    kernel_holder: dict[str, object] = {}
    search_policy = build_record_search_policy(
        lambda: kernel_holder["kernel"].keyword_store,  # type: ignore[union-attr]
        lambda identity: kernel_holder["kernel"].backend.hydrate_record(identity),  # type: ignore[union-attr]
        lambda incoming: kernel_holder["kernel"].pipeline._graph_store.set_direction(  # type: ignore[union-attr]
            incoming
        ),
    )
    kernel = build_local_record_kernel(
        Path(config.indexing.index_path) / "index.db",
        embedding_provider=embedding_provider,
        embedding_model_name=embedding_provider.model_name,
        embedding_dim=embedding_provider.dim,
        vector_engine="exact",
        search_policy=search_policy,
    )
    kernel_holder["kernel"] = kernel
    manager = RecordIndexManager(
        config,
        kernel,
        embedding_provider,
        documents_roots=documents_roots,
    )
    install_bidirectional_graph_store(
        kernel,
        lambda: tuple(
            RecordIdentity.from_storage_key(key)
            for keys in manager._source_records.values()
            for key in keys
        ),
    )
    kernel_holder["manager"] = manager
    return manager


def make_record(
    source_id: str,
    body: str,
    *,
    source_kind: str = "note",
    workspace_id: str | None = None,
    metadata: dict[str, object] | None = None,
) -> Record:
    """Create a canonical record for tests that seed non-file sources."""

    now = datetime.now(UTC)
    return Record(
        workspace_id=workspace_id,
        source_kind=source_kind,
        source_id=source_id,
        title=source_id,
        body=body,
        created_at=now,
        updated_at=now,
        metadata=dict(metadata or {}),
        status=RecordStatus.ACTIVE,
    )
