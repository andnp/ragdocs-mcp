from abc import ABC, abstractmethod

from src.models import Chunk


class ChunkingStrategy(ABC):
    @abstractmethod
    def chunk_document(self, document) -> list[Chunk]:
        pass
