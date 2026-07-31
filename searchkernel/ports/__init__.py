"""Port/Protocol interfaces for the search kernel.

Ports define the contracts between the kernel core and the outside world.
They are purely abstract (Protocol or ABC); implementations live in adapters/.

Dependency rule: ports import only from domain/ and stdlib/typing.
"""

from searchkernel.ports.chunking_config import ChunkTuningConfig
from searchkernel.ports.content_source import ContentSource, SearchableSource
from searchkernel.ports.embedding import (
    EmbeddingBatchProvider,
    EmbeddingProvider,
    EmbeddingSink,
)
from searchkernel.ports.index_manager import IndexManagerPort
from searchkernel.ports.llm import LLMProvider
from searchkernel.ports.orchestrator_config import OrchestratorConfig
from searchkernel.ports.rerank import Reranker
from searchkernel.ports.search import SearchAPI
from searchkernel.ports.stores import CacheStore, GraphStore, KeywordStore, VectorStore

__all__ = [
    "CacheStore",
    "ChunkTuningConfig",
    "ContentSource",
    "EmbeddingBatchProvider",
    "EmbeddingProvider",
    "EmbeddingSink",
    "GraphStore",
    "IndexManagerPort",
    "KeywordStore",
    "LLMProvider",
    "OrchestratorConfig",
    "Reranker",
    "SearchAPI",
    "SearchableSource",
    "VectorStore",
]
