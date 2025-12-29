from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable


@dataclass(slots=True)
class SearchResult:
    doc_id: str
    score: float
    metadata: dict[str, Any]


@runtime_checkable
class IndexProtocol(Protocol):
    def search(self, query: str, limit: int = 10) -> list[SearchResult]: ...

    def add_document(self, doc_id: str, content: str, metadata: dict[str, Any]) -> None: ...

    def remove_document(self, doc_id: str) -> None: ...

    def clear(self) -> None: ...

    def save(self, path: Path) -> None: ...

    def load(self, path: Path) -> None: ...

    def __len__(self) -> int: ...
