"""Core domain types for the search kernel.

Pure data types representing the source-agnostic contracts between the kernel
and the outside world. No I/O, no imports from adapters/runtime/stores.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

# ===== Supporting types =====

class RecordStatus(str, Enum):
    """Lifecycle status of a record in the kernel."""

    ACTIVE = "active"
    STALE = "stale"
    ARCHIVED = "archived"


class Tier(str, Enum):
    """Tier for LLM provider selection (performance vs. quality)."""

    FAST = "fast"      # High-volume, low-latency (SLM, local)
    SMART = "smart"    # Higher quality, higher latency (Claude, etc.)


# Type aliases for clarity in port signatures
Vector = list[float]  # Embedding vector
Cursor = str | None  # Watermark for incremental sync (e.g., commit SHA, timestamp)
Filters = dict[str, Any]  # Query filters (source-specific, opaque to core)
ChangeSignal = dict[str, Any]  # Source change info: {"watch": bool, "poll_interval": int}


# ===== Core domain types =====

@dataclass
class Chunk:
    """A discrete unit of content to be embedded and indexed.

    Chunks are derived from source records during ingestion. They carry
    enough context to reconstruct their parent record and to hydrate results.
    """

    chunk_id: str
    """Unique identifier for this chunk."""

    record_id: str
    """ID of the parent Record this chunk came from."""

    content: str
    """Plain-text chunk content to be embedded."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Chunk-level metadata (e.g., section headers, position in document)."""

    chunk_index: int = 0
    """Position of this chunk within its parent record."""

    content_hash: str = ""
    """SHA256 hash of content for change detection (computed on demand)."""

    def compute_content_hash(self) -> str:
        """Compute SHA256 hash of chunk content for change detection."""
        import hashlib

        return hashlib.sha256(self.content.encode("utf-8")).hexdigest()


@dataclass
class Record:
    """A source-agnostic record representing indexable content.

    Records are the contract between content sources and the kernel. A source
    adapts its native schema into Records; the kernel chunks, embeds, and
    indexes them. Records can carry pre-computed embeddings if the source
    already has them (avoids re-embedding for federated sources).
    """

    source_kind: str
    """
    Source type identifier: "note", "git_commit", "gmail", "jira", etc.
    Determines which adapter produced this record.
    """

    source_id: str
    """
    Stable, namespaced identifier within the source.
    Examples: "git:abc123def456", "gmail:msg-12345", "jira:CORE-999"
    """

    title: str
    """Human-readable title or headline."""

    body: str
    """Main content as plain text (extracted from HTML/Markdown/etc.)."""

    created_at: datetime
    """When the record was originally created in the source."""

    updated_at: datetime
    """Last modified time; used as the watermark for incremental sync."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Source-specific metadata (opaque to the core; preserved in results)."""

    uri: str | None = None
    """Permalink or file path for citation/navigation."""

    status: RecordStatus = RecordStatus.ACTIVE
    """Lifecycle status: active, stale, or archived."""

    embedding: Vector | None = None
    """Pre-computed embedding (if the source brought its own vectors)."""

    embedding_model: str | None = None
    """Model name that produced the embedding (if embedding is set)."""

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a dictionary for storage or RPC."""
        return {
            "source_kind": self.source_kind,
            "source_id": self.source_id,
            "title": self.title,
            "body": self.body,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "metadata": self.metadata,
            "uri": self.uri,
            "status": self.status.value,
            "embedding": self.embedding,
            "embedding_model": self.embedding_model,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Record":
        """Deserialize from a dictionary."""
        from datetime import datetime as dt

        # Parse ISO datetime strings
        created_at = data["created_at"]
        if isinstance(created_at, str):
            created_at = dt.fromisoformat(created_at)

        updated_at = data["updated_at"]
        if isinstance(updated_at, str):
            updated_at = dt.fromisoformat(updated_at)

        # Parse status enum
        status = data.get("status", RecordStatus.ACTIVE)
        if isinstance(status, str):
            status = RecordStatus(status)

        return cls(
            source_kind=data["source_kind"],
            source_id=data["source_id"],
            title=data["title"],
            body=data["body"],
            created_at=created_at,
            updated_at=updated_at,
            metadata=data.get("metadata", {}),
            uri=data.get("uri"),
            status=status,
            embedding=data.get("embedding"),
            embedding_model=data.get("embedding_model"),
        )


@dataclass
class SearchResult:
    """A ranked result from a search query.

    Returned by the SearchAPI; contains the matched record, its score,
    and provenance information showing which stages contributed to the ranking.
    """

    record_id: str
    """ID of the matched Record."""

    score: float
    """Relevance score (normalized across fusion sources)."""

    source_kind: str
    """Source type of the matched record."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Additional metadata (e.g., strategy contributions, adjustments)."""

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a dictionary."""
        return {
            "record_id": self.record_id,
            "score": self.score,
            "source_kind": self.source_kind,
            "metadata": self.metadata,
        }


@dataclass
class ScoredRef:
    """A ranked reference returned by a SearchableSource.

    Used in federation: when a source runs its own retrieval, it returns
    an ordered list of ScoredRefs. The kernel fuses these across sources.
    """

    source_id: str
    """The stable identifier of the matched record in its source."""

    score: float
    """Relevance score (source-specific scale; used for RRF fusion)."""

    source_kind: str
    """Source type (must match the adapter's source_kind)."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Source-specific result metadata."""

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a dictionary."""
        return {
            "source_id": self.source_id,
            "score": self.score,
            "source_kind": self.source_kind,
            "metadata": self.metadata,
        }
