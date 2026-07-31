from abc import ABC, abstractmethod

from searchkernel.models import Document


class DocumentParser(ABC):
    @abstractmethod
    def parse(self, file_path: str) -> Document:
        pass
