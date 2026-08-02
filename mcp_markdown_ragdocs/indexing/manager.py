"""Compatibility import for the canonical record index manager."""

from mcp_markdown_ragdocs.indexing.record_manager import (
    PreparedRecordDocument,
    RecordIndexManager,
    build_embedding_provider,
)

IndexManager = RecordIndexManager

__all__ = [
    "IndexManager",
    "PreparedRecordDocument",
    "RecordIndexManager",
    "build_embedding_provider",
]
