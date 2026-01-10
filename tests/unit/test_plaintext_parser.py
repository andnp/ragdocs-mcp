import pytest

from src.parsers.plaintext import PlainTextParser


def test_parse_simple_text(tmp_path):
    file = tmp_path / "test.txt"
    file.write_text("Hello world\n\nSecond paragraph", encoding="utf-8")

    parser = PlainTextParser()
    doc = parser.parse(str(file))

    assert doc.content == "Hello world\n\nSecond paragraph"
    assert doc.metadata["source"] == str(file)
    assert doc.links == []
    assert doc.tags == []
    assert doc.id == "test"
    assert doc.file_path == str(file)


def test_parse_utf8_encoding(tmp_path):
    file = tmp_path / "utf8.txt"
    file.write_text("Unicode: café, naïve, 日本語", encoding="utf-8")

    parser = PlainTextParser()
    doc = parser.parse(str(file))

    assert "café" in doc.content
    assert "naïve" in doc.content
    assert "日本語" in doc.content
    assert "encoding" not in doc.metadata or doc.metadata["encoding"] == "utf-8"


def test_parse_encoding_fallback(tmp_path):
    file = tmp_path / "latin.txt"
    file.write_bytes(b"Caf\xe9")

    parser = PlainTextParser()
    doc = parser.parse(str(file))

    assert "Caf" in doc.content
    assert doc.metadata.get("encoding") in ["latin-1", "cp1252", "iso-8859-1"]


def test_parse_nonexistent_file():
    parser = PlainTextParser()
    with pytest.raises(FileNotFoundError):
        parser.parse("/nonexistent/file.txt")


def test_parse_empty_file(tmp_path):
    file = tmp_path / "empty.txt"
    file.write_text("", encoding="utf-8")

    parser = PlainTextParser()
    doc = parser.parse(str(file))

    assert doc.content == ""
    assert doc.id == "empty"


def test_parse_multiline_content(tmp_path):
    file = tmp_path / "multi.txt"
    content = "Line 1\nLine 2\n\nParagraph 2\n\nParagraph 3"
    file.write_text(content, encoding="utf-8")

    parser = PlainTextParser()
    doc = parser.parse(str(file))

    assert doc.content == content


def test_parse_preserves_whitespace(tmp_path):
    file = tmp_path / "whitespace.txt"
    content = "  Leading spaces\n\tTab character\nTrailing spaces  "
    file.write_text(content, encoding="utf-8")

    parser = PlainTextParser()
    doc = parser.parse(str(file))

    assert doc.content == content


def test_parse_metadata_structure(tmp_path):
    file = tmp_path / "test.txt"
    file.write_text("content", encoding="utf-8")

    parser = PlainTextParser()
    doc = parser.parse(str(file))

    assert "source" in doc.metadata
    assert isinstance(doc.metadata["source"], str)
    assert doc.modified_time is not None
