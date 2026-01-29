from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class MemoryFrontmatter:
    type: str = "journal"
    status: str = "active"
    tags: list[str] = field(default_factory=list)
    created_at: datetime | None = None

    def __post_init__(self):
        valid_types = ("plan", "journal", "fact", "observation", "reflection")
        if self.type not in valid_types:
            self.type = "journal"

        valid_statuses = ("active", "archived")
        if self.status not in valid_statuses:
            self.status = "active"


@dataclass
class MemorySearchStats:
    """Statistics about memory search filtering stages.

    Provides visibility into where memories are being filtered out during search,
    helping diagnose issues with search quality and thresholds.
    """
    total_indexed: int = 0  # Total memories in the index
    vector_candidates: int = 0  # Results from vector search
    keyword_candidates: int = 0  # Results from keyword search
    after_fusion: int = 0  # Unique results after RRF fusion
    filtered_missing_chunk: int = 0  # Filtered due to missing chunk data
    filtered_type_mismatch: int = 0  # Filtered due to type filter
    filtered_time_range: int = 0  # Filtered due to time range
    filtered_below_threshold: int = 0  # Filtered due to score < threshold
    score_threshold: float = 0.0  # The threshold used
    min_score_seen: float = 1.0  # Minimum score seen (helps diagnose threshold issues)
    max_score_seen: float = 0.0  # Maximum score seen
    returned: int = 0  # Final count returned (after limit)


@dataclass
class ExtractedLink:
    target: str
    edge_type: str
    anchor_context: str
    position: int
    is_memory_link: bool = False


@dataclass
class MemoryDocument:
    id: str
    content: str
    frontmatter: MemoryFrontmatter
    links: list[ExtractedLink]
    file_path: str
    modified_time: datetime


@dataclass
class MemorySearchResult:
    memory_id: str
    score: float
    content: str
    frontmatter: MemoryFrontmatter
    file_path: str
    header_path: str = ""


@dataclass
class LinkedMemoryResult:
    memory_id: str
    score: float
    content: str
    anchor_context: str
    edge_type: str
    file_path: str
