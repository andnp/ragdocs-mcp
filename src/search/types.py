from typing import TypedDict


class SearchResultDict(TypedDict, total=False):
    chunk_id: str
    doc_id: str
    content: str
    score: float
    file_path: str
    header_path: str
    metadata: dict
