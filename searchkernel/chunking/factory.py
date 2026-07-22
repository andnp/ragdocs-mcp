from searchkernel.chunking.base import ChunkingStrategy
from searchkernel.chunking.header_chunker import HeaderBasedChunker
from searchkernel.config import ChunkingConfig


def get_chunker(config: ChunkingConfig) -> ChunkingStrategy:
    return HeaderBasedChunker(config)
