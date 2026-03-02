from typing import NotRequired, TypedDict


class SearchResultDict(TypedDict):
    chunk_id: str
    doc_id: str
    content: NotRequired[str]
    score: NotRequired[float]
    file_path: NotRequired[str]
    header_path: NotRequired[str]
    metadata: NotRequired[dict]
