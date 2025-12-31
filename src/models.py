from dataclasses import dataclass
from datetime import datetime


@dataclass
class CodeBlock:
    id: str
    doc_id: str
    chunk_id: str
    content: str
    language: str


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


@dataclass
class ChunkResult:
    chunk_id: str
    doc_id: str
    score: float
    header_path: str
    file_path: str
    content: str = ""
    parent_chunk_id: str | None = None
    parent_content: str | None = None

    def to_dict(self):
        result = {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "score": self.score,
            "header_path": self.header_path,
            "file_path": self.file_path,
            "content": self.content,
        }
        if self.parent_chunk_id is not None:
            result["parent_chunk_id"] = self.parent_chunk_id
        if self.parent_content is not None:
            result["parent_content"] = self.parent_content
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
class Document:
    id: str
    content: str
    metadata: dict[str, str | list[str] | int | float | bool]
    links: list[str]
    tags: list[str]
    file_path: str
    modified_time: datetime
    chunks: list[Chunk] | None = None
