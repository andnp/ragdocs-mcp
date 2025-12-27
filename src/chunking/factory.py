from src.chunking.base import ChunkingStrategy
from src.chunking.header_chunker import HeaderBasedChunker
from src.config import ChunkingConfig


def get_chunker(config: ChunkingConfig) -> ChunkingStrategy:
    if config.strategy == "header_based":
        return HeaderBasedChunker(config)
    elif config.strategy == "none":
        raise ValueError("Chunking disabled (strategy='none')")
    else:
        raise ValueError(f"Unknown chunking strategy: {config.strategy}")
