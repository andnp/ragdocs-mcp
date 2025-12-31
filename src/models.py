from dataclasses import dataclass
from datetime import datetime


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


@dataclass
class ChunkResult:
    chunk_id: str
    doc_id: str
    score: float
    header_path: str
    file_path: str
    content: str = ""

    def to_dict(self):
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "score": self.score,
            "header_path": self.header_path,
            "file_path": self.file_path,
            "content": self.content,
        }


@dataclass
class CompressionStats:
    original_count: int
    after_threshold: int
    after_content_dedup: int
    after_dedup: int
    after_doc_limit: int
    clusters_merged: int

    def to_dict(self):
        return {
            "original_count": self.original_count,
            "after_threshold": self.after_threshold,
            "after_content_dedup": self.after_content_dedup,
            "after_dedup": self.after_dedup,
            "after_doc_limit": self.after_doc_limit,
            "clusters_merged": self.clusters_merged,
        }


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
