from abc import ABC, abstractmethod

from mcp_markdown_ragdocs.models import Document


class DocumentParser(ABC):
    @abstractmethod
    def parse(self, file_path: str) -> Document:
        pass
