"""Canonical record-kernel helpers shared by integration tests."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from searchkernel.api import build_local_record_kernel
from searchkernel.domain import Record, RecordStatus

from mcp_markdown_ragdocs.config import Config
from mcp_markdown_ragdocs.indexing.record_manager import (
    RecordIndexManager,
    build_embedding_provider,
)


def make_record_index_manager(
    config: Config,
    *,
    documents_roots: list[Path] | None = None,
) -> RecordIndexManager:
    """Build the same local record stack used by the application context."""

    model_name = config.embedding.model_name
    embedding_provider = build_embedding_provider(config, model_name)
    kernel = build_local_record_kernel(
        Path(config.indexing.index_path) / "index.db",
        embedding_provider=embedding_provider,
        embedding_model_name=embedding_provider.model_name,
        embedding_dim=embedding_provider.dim,
        vector_engine="exact",
    )
    return RecordIndexManager(
        config,
        kernel,
        embedding_provider,
        documents_roots=documents_roots,
    )


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
