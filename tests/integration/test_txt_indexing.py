import pytest

from src.config import Config, IndexingConfig, LLMConfig, SearchConfig, ServerConfig, ChunkingConfig
from src.indexing.manager import IndexManager
from src.indices.graph import GraphStore
from src.indices.keyword import KeywordIndex
from src.indices.vector import VectorIndex
from src.parsers.plaintext import PlainTextParser
from src.chunking.header_chunker import HeaderBasedChunker


@pytest.fixture
def config(tmp_path):
    docs_path = tmp_path / "docs"
    docs_path.mkdir()
    return Config(
        server=ServerConfig(),
        indexing=IndexingConfig(
            documents_path=str(docs_path),
            index_path=str(tmp_path / "indices"),
        ),
        parsers={
            "**/*.md": "MarkdownParser",
            "**/*.txt": "PlainTextParser"
        },
        search=SearchConfig(
            semantic_weight=1.0,
            keyword_weight=1.0,
            recency_bias=0.5,
            rrf_k_constant=60,
        ),
        llm=LLMConfig(embedding_model="all-MiniLM-L6-v2"),
        chunking=ChunkingConfig(
            min_chunk_chars=200,
            max_chunk_chars=2000,
            overlap_chars=100,
        ),
    )


@pytest.fixture
def indices():
    vector = VectorIndex()
    keyword = KeywordIndex()
    graph = GraphStore()
    return vector, keyword, graph


@pytest.fixture
def manager(config, indices):
    vector, keyword, graph = indices
    return IndexManager(config, vector, keyword, graph)


def test_index_txt_file(tmp_path, manager):
    txt_file = tmp_path / "docs" / "notes.txt"
    txt_file.write_text(
        "Machine Learning Notes\n\n"
        "Neural networks are computational models inspired by biological brains.\n\n"
        "Training requires large amounts of labeled data."
    )

    manager.index_document(str(txt_file))

    results = manager.vector.search("neural networks", top_k=5)

    assert len(results) > 0
    assert any("neural networks" in r["content"].lower() for r in results if "content" in r)


def test_txt_chunking_respects_size_limits(tmp_path, config):
    txt_file = tmp_path / "docs" / "large.txt"
    paragraphs = [f"This is paragraph number {i}. " * 50 for i in range(50)]
    txt_file.write_text("\n\n".join(paragraphs))

    parser = PlainTextParser()
    doc = parser.parse(str(txt_file))

    chunker = HeaderBasedChunker(config.chunking)
    chunks = chunker.chunk_document(doc)

    for chunk in chunks:
        assert len(chunk.content) >= config.chunking.min_chunk_chars
        assert len(chunk.content) <= config.chunking.max_chunk_chars


def test_txt_chunks_have_no_header_path(tmp_path, config):
    txt_file = tmp_path / "docs" / "plain.txt"
    txt_file.write_text("Paragraph 1\n\nParagraph 2\n\nParagraph 3")

    parser = PlainTextParser()
    doc = parser.parse(str(txt_file))

    chunker = HeaderBasedChunker(config.chunking)
    chunks = chunker.chunk_document(doc)

    for chunk in chunks:
        assert chunk.header_path == ""


def test_txt_small_content_single_chunk(tmp_path, config):
    txt_file = tmp_path / "docs" / "small.txt"
    txt_file.write_text("This is a small text file with minimal content.")

    parser = PlainTextParser()
    doc = parser.parse(str(txt_file))

    chunker = HeaderBasedChunker(config.chunking)
    chunks = chunker.chunk_document(doc)

    assert len(chunks) == 1
    assert chunks[0].content == "This is a small text file with minimal content."


def test_search_retrieves_txt_chunks(tmp_path, manager):
    txt_file = tmp_path / "docs" / "database.txt"
    txt_file.write_text(
        "Database Systems Overview\n\n"
        "Relational databases use SQL for querying structured data.\n\n"
        "NoSQL databases handle unstructured or semi-structured data efficiently."
    )

    manager.index_document(str(txt_file))

    results = manager.vector.search("SQL relational database", top_k=5)

    assert len(results) > 0
    found = False
    for result in results:
        if "content" in result and "sql" in result["content"].lower():
            found = True
            assert result["header_path"] == ""
            assert "database" in result["file_path"].lower()
            break
    assert found


def test_mixed_md_and_txt_indexing(tmp_path, manager):
    md_file = tmp_path / "docs" / "readme.md"
    md_file.write_text("# Project\n\nMarkdown content here.")

    txt_file = tmp_path / "docs" / "notes.txt"
    txt_file.write_text("Plain text content here.")

    manager.index_document(str(md_file))
    manager.index_document(str(txt_file))

    md_results = manager.vector.search("markdown", top_k=5)
    assert any("markdown" in r["content"].lower() for r in md_results if "content" in r)

    txt_results = manager.vector.search("plain text", top_k=5)
    assert any("plain text" in r["content"].lower() for r in txt_results if "content" in r)


def test_txt_chunk_start_end_positions(tmp_path, config):
    txt_file = tmp_path / "docs" / "positions.txt"
    content = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
    txt_file.write_text(content)

    parser = PlainTextParser()
    doc = parser.parse(str(txt_file))

    chunker = HeaderBasedChunker(config.chunking)
    chunks = chunker.chunk_document(doc)

    for chunk in chunks:
        assert chunk.start_pos >= 0
        assert chunk.end_pos <= len(content)
        assert chunk.start_pos < chunk.end_pos


def test_txt_multiple_paragraphs_chunking(tmp_path, config):
    txt_file = tmp_path / "docs" / "multi.txt"
    paragraphs = [f"Paragraph {i} with some content." for i in range(10)]
    txt_file.write_text("\n\n".join(paragraphs))

    parser = PlainTextParser()
    doc = parser.parse(str(txt_file))

    chunker = HeaderBasedChunker(config.chunking)
    chunks = chunker.chunk_document(doc)

    assert len(chunks) >= 1

    original_content = doc.content

    for para in paragraphs:
        assert para in original_content


def test_txt_unicode_content_search(tmp_path, manager):
    txt_file = tmp_path / "docs" / "unicode.txt"
    txt_file.write_text(
        "International Characters\n\n"
        "This file contains café, naïve, and 日本語 characters."
    )

    manager.index_document(str(txt_file))

    results = manager.vector.search("international characters", top_k=5)
    assert len(results) > 0


def test_txt_empty_paragraphs_handled(tmp_path, config):
    txt_file = tmp_path / "docs" / "empty.txt"
    txt_file.write_text("Para 1\n\n\n\nPara 2\n\n\n\n\n\nPara 3")

    parser = PlainTextParser()
    doc = parser.parse(str(txt_file))

    chunker = HeaderBasedChunker(config.chunking)
    chunks = chunker.chunk_document(doc)

    assert len(chunks) >= 1
    for chunk in chunks:
        assert chunk.content.strip()


def test_txt_metadata_preserved(tmp_path, config):
    txt_file = tmp_path / "docs" / "meta.txt"
    txt_file.write_text("Content with metadata.")

    parser = PlainTextParser()
    doc = parser.parse(str(txt_file))

    chunker = HeaderBasedChunker(config.chunking)
    chunks = chunker.chunk_document(doc)

    for chunk in chunks:
        assert chunk.file_path == str(txt_file)
        assert chunk.modified_time == doc.modified_time
        assert chunk.doc_id == doc.id


def test_txt_chunk_ids_unique(tmp_path, config):
    txt_file = tmp_path / "docs" / "test.txt"
    content = "\n\n".join([f"Paragraph {i}" * 100 for i in range(10)])
    txt_file.write_text(content)

    parser = PlainTextParser()
    doc = parser.parse(str(txt_file))

    chunker = HeaderBasedChunker(config.chunking)
    chunks = chunker.chunk_document(doc)

    chunk_ids = [c.chunk_id for c in chunks]
    assert len(chunk_ids) == len(set(chunk_ids))
