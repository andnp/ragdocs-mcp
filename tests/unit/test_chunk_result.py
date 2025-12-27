"""
Unit tests for ChunkResult dataclass.

Tests the ChunkResult model serialization, validation, and data integrity.
"""

from src.models import ChunkResult


def test_chunk_result_creation():
    """
    Verify ChunkResult dataclass can be instantiated with all fields.

    Tests that the dataclass accepts all required fields and stores them correctly.
    """
    result = ChunkResult(
        chunk_id="README_chunk_0",
        doc_id="README",
        score=0.95,
        header_path="Introduction > Getting Started",
        file_path="docs/README.md",
    )

    assert result.chunk_id == "README_chunk_0"
    assert result.doc_id == "README"
    assert result.score == 0.95
    assert result.header_path == "Introduction > Getting Started"
    assert result.file_path == "docs/README.md"


def test_chunk_result_to_dict():
    """
    Verify to_dict() returns correct dictionary structure.

    Tests that serialization produces the expected JSON-compatible dictionary
    with all required fields.
    """
    result = ChunkResult(
        chunk_id="test_chunk_1",
        doc_id="test_doc",
        score=1.0,
        header_path="Section A > Subsection B",
        file_path="path/to/file.md",
    )

    result_dict = result.to_dict()

    assert isinstance(result_dict, dict)
    assert result_dict["chunk_id"] == "test_chunk_1"
    assert result_dict["doc_id"] == "test_doc"
    assert result_dict["score"] == 1.0
    assert result_dict["header_path"] == "Section A > Subsection B"
    assert result_dict["file_path"] == "path/to/file.md"
    assert result_dict["content"] == ""

    # Verify all expected keys are present
    expected_keys = {"chunk_id", "doc_id", "score", "header_path", "file_path", "content"}
    assert set(result_dict.keys()) == expected_keys


def test_chunk_result_to_dict_types():
    """
    Verify dict values have correct types (str, float).

    Tests that serialization preserves proper data types for JSON compatibility.
    """
    result = ChunkResult(
        chunk_id="chunk_123",
        doc_id="doc_456",
        score=0.75,
        header_path="Main Header",
        file_path="file.md",
    )

    result_dict = result.to_dict()

    assert isinstance(result_dict["chunk_id"], str)
    assert isinstance(result_dict["doc_id"], str)
    assert isinstance(result_dict["score"], float)
    assert isinstance(result_dict["header_path"], str)
    assert isinstance(result_dict["file_path"], str)
    assert isinstance(result_dict["content"], str)


def test_chunk_result_score_range_valid():
    """
    Verify valid score values are accepted.

    Tests that scores within [0.0, 1.0] range are properly stored.
    """
    # Test boundary values
    result_zero = ChunkResult(
        chunk_id="test_1",
        doc_id="test",
        score=0.0,
        header_path="",
        file_path="",
    )
    assert result_zero.score == 0.0

    result_one = ChunkResult(
        chunk_id="test_2",
        doc_id="test",
        score=1.0,
        header_path="",
        file_path="",
    )
    assert result_one.score == 1.0

    # Test mid-range value
    result_mid = ChunkResult(
        chunk_id="test_3",
        doc_id="test",
        score=0.5,
        header_path="",
        file_path="",
    )
    assert result_mid.score == 0.5


def test_chunk_result_with_empty_metadata():
    """
    Verify ChunkResult handles empty metadata fields gracefully.

    Tests fallback behavior when header_path or file_path are empty strings.
    """
    result = ChunkResult(
        chunk_id="chunk_0",
        doc_id="doc_0",
        score=0.8,
        header_path="",
        file_path="",
    )

    assert result.header_path == ""
    assert result.file_path == ""

    result_dict = result.to_dict()
    assert result_dict["header_path"] == ""
    assert result_dict["file_path"] == ""


def test_chunk_result_equality():
    """
    Verify ChunkResult instances can be compared for equality.

    Tests that dataclass equality works as expected (compares all fields).
    """
    result1 = ChunkResult(
        chunk_id="chunk_1",
        doc_id="doc_1",
        score=0.9,
        header_path="Header A",
        file_path="path/a.md",
    )

    result2 = ChunkResult(
        chunk_id="chunk_1",
        doc_id="doc_1",
        score=0.9,
        header_path="Header A",
        file_path="path/a.md",
    )

    result3 = ChunkResult(
        chunk_id="chunk_2",
        doc_id="doc_1",
        score=0.9,
        header_path="Header A",
        file_path="path/a.md",
    )

    assert result1 == result2
    assert result1 != result3


def test_chunk_result_repr():
    """
    Verify ChunkResult has a useful string representation.

    Tests that the dataclass provides informative __repr__ output.
    """
    result = ChunkResult(
        chunk_id="test_chunk",
        doc_id="test_doc",
        score=0.85,
        header_path="Test Header",
        file_path="test.md",
    )

    repr_str = repr(result)

    # Verify key fields appear in repr
    assert "test_chunk" in repr_str
    assert "test_doc" in repr_str
    assert "0.85" in repr_str
