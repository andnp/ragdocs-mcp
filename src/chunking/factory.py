from src.chunking.base import ChunkingStrategy
from src.chunking.header_chunker import HeaderBasedChunker
from src.config import ChunkingConfig


def get_chunker(config: ChunkingConfig) -> ChunkingStrategy:
    return HeaderBasedChunker(config)
