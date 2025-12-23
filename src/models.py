from dataclasses import dataclass
from datetime import datetime


@dataclass
class Document:
    id: str
    content: str
    metadata: dict[str, str | list[str] | int | float | bool]
    links: list[str]
    tags: list[str]
    file_path: str
    modified_time: datetime
