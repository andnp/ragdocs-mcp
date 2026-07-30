from searchkernel.chunking.base import ChunkingStrategy
from searchkernel.chunking.header_chunker import HeaderBasedChunker
from searchkernel.ports.chunking_config import ChunkTuningConfig


def get_chunker(config: ChunkTuningConfig) -> ChunkingStrategy:
    return HeaderBasedChunker(config)
