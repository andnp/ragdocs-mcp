from abc import ABC, abstractmethod

from src.models import Document


class DocumentParser(ABC):
    @abstractmethod
    def parse(self, file_path: str) -> Document:
        pass
