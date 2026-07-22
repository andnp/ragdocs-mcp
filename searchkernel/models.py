from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import hashlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from searchkernel.domain import Record


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


@dataclass(frozen=True)
class StrategyContribution:
    rank: int
    raw_score: float

    def to_dict(self):
        return {
            "rank": self.rank,
            "raw_score": self.raw_score,
        }


@dataclass
class SearchResultProvenance:
    strategies: tuple[str, ...] = ()
    strategy_details: dict[str, StrategyContribution] = field(default_factory=dict)
    community_boost: float | None = None
    project_uplift: float | None = None
    parent_expanded_from: str | None = None

    def add_strategy(self, strategy: str, rank: int, raw_score: float):
        if strategy in self.strategy_details:
            return

        self.strategy_details[strategy] = StrategyContribution(
            rank=rank,
            raw_score=raw_score,
        )
        if strategy not in self.strategies:
            self.strategies = (*self.strategies, strategy)

    def clone(self):
        return SearchResultProvenance(
            strategies=tuple(self.strategies),
            strategy_details=dict(self.strategy_details),
            community_boost=self.community_boost,
            project_uplift=self.project_uplift,
            parent_expanded_from=self.parent_expanded_from,
        )

    def to_dict(self):
        result: dict[str, object] = {
            "strategies": list(self.strategies),
        }
        if self.strategy_details:
            result["strategy_details"] = {
                strategy: contribution.to_dict()
                for strategy, contribution in self.strategy_details.items()
            }

        adjustments: dict[str, object] = {}
        if self.community_boost is not None and self.community_boost != 1.0:
            adjustments["community_boost"] = self.community_boost
        if self.project_uplift is not None and self.project_uplift != 1.0:
            adjustments["project_uplift"] = self.project_uplift
        if self.parent_expanded_from is not None:
            adjustments["parent_expanded_from"] = self.parent_expanded_from
        if adjustments:
            result["adjustments"] = adjustments

        return result


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


@dataclass
class CompressionStats:
    original_count: int
    after_threshold: int
    after_content_dedup: int
    after_ngram_dedup: int
    after_dedup: int
    after_doc_limit: int
    clusters_merged: int

    def to_dict(self):
        return {
            "original_count": self.original_count,
            "after_threshold": self.after_threshold,
            "after_content_dedup": self.after_content_dedup,
            "after_ngram_dedup": self.after_ngram_dedup,
            "after_dedup": self.after_dedup,
            "after_doc_limit": self.after_doc_limit,
            "clusters_merged": self.clusters_merged,
        }


@dataclass
class SearchStrategyStats:
    vector_count: int | None = None
    keyword_count: int | None = None
    graph_count: int | None = None
    tag_expansion_count: int | None = None

    def to_dict(self):
        result = {}
        if self.vector_count is not None:
            result["vector_count"] = self.vector_count
        if self.keyword_count is not None:
            result["keyword_count"] = self.keyword_count
        if self.graph_count is not None:
            result["graph_count"] = self.graph_count
        if self.tag_expansion_count is not None:
            result["tag_expansion_count"] = self.tag_expansion_count
        return result


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

    def to_record(self) -> "Record":
        """Convert a Document to a domain Record."""
        from searchkernel.domain import Record, RecordStatus

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
    def from_record(cls, record: "Record") -> "Document":
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
        from searchkernel.domain import Record, RecordStatus
        from datetime import datetime, timezone

        # Create timestamp from Unix seconds
        created_at = datetime.fromtimestamp(self.timestamp, tz=timezone.utc)

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
