from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import UTC, datetime
from searchkernel.api import (
    ChunkResult as DomainChunkResult,
    CompressionStats,
    Record,
    RecordStatus,
    SearchResultProvenance,
    SearchStrategyStats,
    StrategyContribution,
)

__all__ = [
    "Chunk",
    "ChunkResult",
    "CommitResult",
    "CompressionStats",
    "Document",
    "GitSearchResponse",
    "ReconciliationResult",
    "SearchResultProvenance",
    "SearchStrategyStats",
    "StrategyContribution",
]


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    content: str
    metadata: dict
    chunk_index: int
    header_path: str
    start_pos: int
    end_pos: int
    file_path: str
    modified_time: datetime
    parent_chunk_id: str | None = None
    project_id: str | None = None
    content_hash: str = field(default="", init=False)

    def __post_init__(self):
        """Compute content hash after initialization."""
        if not self.content_hash:
            self.content_hash = self.compute_content_hash()

    def compute_content_hash(self) -> str:
        """Compute SHA256 hash of chunk content for change detection."""
        return hashlib.sha256(self.content.encode("utf-8")).hexdigest()


@dataclass
class ChunkResult:
    chunk_id: str
    doc_id: str
    score: float
    header_path: str
    file_path: str
    project_id: str | None = None
    content: str = ""
    parent_chunk_id: str | None = None
    parent_content: str | None = None
    provenance: SearchResultProvenance | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    def to_dict(self):
        result = {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "score": self.score,
            "header_path": self.header_path,
            "file_path": self.file_path,
            "content": self.content,
        }
        if self.project_id is not None:
            result["project_id"] = self.project_id
        if self.parent_chunk_id is not None:
            result["parent_chunk_id"] = self.parent_chunk_id
        if self.parent_content is not None:
            result["parent_content"] = self.parent_content
        if self.provenance is not None:
            result["provenance"] = self.provenance.to_dict()
        return result

    def to_domain(self) -> DomainChunkResult:
        """Convert a ChunkResult to a domain ChunkResult."""
        metadata = {
            "header_path": self.header_path,
            "file_path": self.file_path,
            **self.metadata,
        }
        if self.project_id is not None:
            metadata["project_id"] = self.project_id

        return DomainChunkResult(
            chunk_id=self.chunk_id,
            record_id=self.doc_id,
            score=self.score,
            content=self.content,
            parent_chunk_id=self.parent_chunk_id,
            parent_content=self.parent_content,
            provenance=self.provenance,
            metadata=metadata,
        )

    @classmethod
    def from_domain(cls, result: DomainChunkResult) -> ChunkResult:
        """Construct a ChunkResult from a domain ChunkResult."""
        metadata = dict(result.metadata)
        header_path = metadata.pop("header_path", "")
        file_path = metadata.pop("file_path", "")
        project_id = metadata.pop("project_id", None)

        return cls(
            chunk_id=result.chunk_id,
            doc_id=result.record_id,
            score=result.score,
            header_path=header_path,
            file_path=file_path,
            project_id=project_id,
            content=result.content,
            parent_chunk_id=result.parent_chunk_id,
            parent_content=result.parent_content,
            provenance=result.provenance,
            metadata=metadata,
        )


@dataclass
class Document:
    id: str
    content: str
    metadata: dict[str, str | list[str] | int | float | bool]
    links: list[str]
    tags: list[str]
    file_path: str
    modified_time: datetime
    chunks: list[Chunk] | None = None
    project_id: str | None = None

    def to_record(self) -> Record:
        """Convert a Document to a domain Record."""
        return Record(
            source_kind="note",
            source_id=f"note:{self.id}",
            title=self.id,  # Use doc ID as fallback title
            body=self.content,
            created_at=self.modified_time,
            updated_at=self.modified_time,
            metadata={
                "links": self.links,
                "tags": self.tags,
                "file_path": self.file_path,
                "project_id": self.project_id,
                **self.metadata,
            },
            uri=f"file://{self.file_path}",
            status=RecordStatus.ACTIVE,
        )

    @classmethod
    def from_record(cls, record: Record) -> Document:
        """Construct a Document from a domain Record."""
        # Extract metadata that Document expects
        metadata = dict(record.metadata)
        links = metadata.pop("links", [])
        tags = metadata.pop("tags", [])
        file_path = metadata.pop("file_path", "")
        project_id = metadata.pop("project_id", None)

        return cls(
            id=record.source_id.replace("note:", ""),
            content=record.body,
            metadata=metadata,
            links=links,
            tags=tags,
            file_path=file_path,
            modified_time=record.updated_at,
            project_id=project_id,
        )


@dataclass
class CommitResult:
    """Git commit search result."""

    hash: str
    title: str
    author: str
    committer: str
    timestamp: int
    message: str
    files_changed: list[str]
    delta_truncated: str
    score: float
    repo_path: str

    def to_record(self) -> Record:
        """Convert a CommitResult to a domain Record."""
        from datetime import datetime

        # Create timestamp from Unix seconds
        created_at = datetime.fromtimestamp(self.timestamp, tz=UTC)

        # Combine title and message for body
        body = f"{self.title}\n\n{self.message}\n\nDelta:\n{self.delta_truncated}"

        return Record(
            source_kind="git_commit",
            source_id=f"git:{self.hash}",
            title=self.title,
            body=body,
            created_at=created_at,
            updated_at=created_at,
            metadata={
                "author": self.author,
                "committer": self.committer,
                "files_changed": self.files_changed,
                "repo_path": self.repo_path,
            },
            uri=f"{self.repo_path}#commit/{self.hash}",
            status=RecordStatus.ACTIVE,
        )

    @classmethod
    def from_record(cls, record: Record, score: float = 0.0) -> CommitResult:
        """Construct a CommitResult from a domain Record."""
        # Extract git-specific metadata
        metadata = record.metadata
        author = metadata.get("author", "Unknown")
        committer = metadata.get("committer", "Unknown")
        files_changed = metadata.get("files_changed", [])
        repo_path = metadata.get("repo_path", "")

        # Parse source_id to extract hash
        hash_val = record.source_id.replace("git:", "")

        # Convert datetime to timestamp
        timestamp = int(record.created_at.timestamp())

        # Extract message from body (first line is title, rest is message)
        body_lines = record.body.split("\n")
        title = record.title
        message = "\n".join(body_lines[1:]) if len(body_lines) > 1 else record.body
        delta_truncated = ""

        # If body contains "Delta:" section, extract it
        if "Delta:" in message:
            parts = message.split("Delta:")
            message = parts[0].strip()
            delta_truncated = parts[1].strip() if len(parts) > 1 else ""

        return cls(
            hash=hash_val,
            title=title,
            author=author,
            committer=committer,
            timestamp=timestamp,
            message=message,
            files_changed=files_changed,
            delta_truncated=delta_truncated,
            score=score,
            repo_path=repo_path,
        )


@dataclass
class GitSearchResponse:
    """Response from git history search."""

    results: list[CommitResult]
    query: str
    total_commits_indexed: int


@dataclass
class ReconciliationResult:
    """Result of index reconciliation operation."""

    added_count: int = 0
    removed_count: int = 0
    moved_count: int = 0
    failed_count: int = 0
