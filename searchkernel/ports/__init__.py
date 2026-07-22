"""Port/Protocol interfaces for the search kernel.

Ports define the contracts between the kernel core and the outside world.
They are purely abstract (Protocol or ABC); implementations live in adapters/.

Dependency rule: ports import only from domain/ and stdlib/typing.
"""

from searchkernel.ports.content_source import ContentSource, SearchableSource
from searchkernel.ports.embedding import EmbeddingProvider
from searchkernel.ports.llm import LLMProvider
from searchkernel.ports.stores import VectorStore, KeywordStore, GraphStore, CacheStore
from searchkernel.ports.search import SearchAPI

__all__ = [
    "ContentSource",
    "SearchableSource",
    "EmbeddingProvider",
    "LLMProvider",
    "VectorStore",
    "KeywordStore",
    "GraphStore",
    "CacheStore",
    "SearchAPI",
]
