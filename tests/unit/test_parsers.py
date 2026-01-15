from datetime import datetime

import pytest

from src.models import Document
from src.parsers.base import DocumentParser


class ConcreteTestParser(DocumentParser):
    """
    Minimal concrete parser for testing the abstract interface.
    """

    def parse(self, file_path: str) -> Document:
        """
        Returns a simple test document.
        """
        return Document(
            id="test-id",
            content="test content",
            metadata={"key": "value"},
            links=[],
            tags=["test"],
            file_path=file_path,
            modified_time=datetime.now(),
        )


def test_document_dataclass_instantiation():
    """
    Verify Document dataclass can be instantiated with all required fields.
    Ensures the data model is correctly defined and accessible.
    """
    doc = Document(
        id="doc-123",
        content="Sample content",
        metadata={"author": "test", "count": 42, "active": True},
        links=["link1", "link2"],
        tags=["tag1", "tag2"],
        file_path="/path/to/file.md",
        modified_time=datetime(2025, 12, 22, 10, 30),
    )

    assert doc.id == "doc-123"
    assert doc.content == "Sample content"
    assert doc.metadata["author"] == "test"
    assert doc.metadata["count"] == 42
    assert doc.metadata["active"] is True
    assert doc.links == ["link1", "link2"]
    assert doc.tags == ["tag1", "tag2"]
    assert doc.file_path == "/path/to/file.md"
    assert doc.modified_time.year == 2025


def test_document_parser_abstract_class():
    """
    Verify DocumentParser cannot be instantiated directly (abstract).
    Enforces proper inheritance pattern for parser implementations.
    """
    with pytest.raises(TypeError, match="Can't instantiate abstract class"):
        DocumentParser()  # pyright: ignore[reportAbstractUsage]


def test_concrete_parser_implements_parse():
    """
    Verify a concrete parser subclass can implement parse() method.
    Demonstrates the contract that concrete parsers must fulfill.
    """
    parser = ConcreteTestParser()
    doc = parser.parse("/test/path.md")

    assert isinstance(doc, Document)
    assert doc.file_path == "/test/path.md"
    assert doc.content == "test content"
    assert doc.id == "test-id"
    assert "test" in doc.tags
