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
