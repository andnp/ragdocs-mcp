import pytest
from searchkernel.chunking.header_chunker import HeaderBasedChunker
from searchkernel.domain import Record

from mcp_markdown_ragdocs.config import (
    ChunkingConfig,
    Config,
    IndexingConfig,
    LLMConfig,
    SearchConfig,
)
from mcp_markdown_ragdocs.models import Document
from mcp_markdown_ragdocs.parsers.plaintext import PlainTextParser
from tests.integration._canonical import make_record_index_manager


def _to_record(doc: Document) -> Record:
    """Adapt a parser-produced models.Document into a domain.Record for chunk_record()."""
    return Record(
        source_kind="note",
        source_id=doc.id,
        title=doc.id,
        body=doc.content,
        created_at=doc.modified_time,
        updated_at=doc.modified_time,
        metadata={"links": doc.links, "tags": doc.tags, "file_path": doc.file_path, **doc.metadata},
        uri=f"file://{doc.file_path}",
    )


@pytest.fixture
def config(tmp_path):
    docs_path = tmp_path / "docs"
    docs_path.mkdir()
    return Config(
        indexing=IndexingConfig(
            documents_path=str(docs_path), index_path=str(tmp_path / "indices")
        ),
        search=SearchConfig(semantic_weight=1.0, keyword_weight=1.0, recency_bias=0.5),
        llm=LLMConfig(embedding_model="all-MiniLM-L6-v2"),
        chunking=ChunkingConfig(
            min_chunk_chars=200, max_chunk_chars=2000, overlap_chars=100
        ),
    )


@pytest.fixture
def manager(config):
    return make_record_index_manager(config)


def _search_records(manager, query):
    return [
        manager.storage.hydrate_record(hit.storage_key)
        for hit in manager.keyword.search(query, 5)
    ]


def test_index_txt_file(tmp_path, manager):
    txt_file = tmp_path / "docs" / "notes.txt"
    txt_file.write_text(
        "Machine Learning Notes\n\n"
        "Neural networks are computational models inspired by biological brains.\n\n"
        "Training requires large amounts of labeled data."
    )

    manager.index_document(str(txt_file))

    results = _search_records(manager, "neural networks")

    assert len(results) > 0
    assert any(
        "neural networks" in r.body.lower() for r in results if r is not None
    )


def test_txt_chunking_respects_size_limits(tmp_path, config):
    txt_file = tmp_path / "docs" / "large.txt"
    paragraphs = [f"This is paragraph number {i}. " * 50 for i in range(50)]
    txt_file.write_text("\n\n".join(paragraphs))

    parser = PlainTextParser()
    doc = parser.parse(str(txt_file))

    chunker = HeaderBasedChunker(config.chunking)
    chunks = chunker.chunk_record(_to_record(doc))

    for chunk in chunks:
        assert len(chunk.content) >= config.chunking.min_chunk_chars
        assert len(chunk.content) <= config.chunking.max_chunk_chars


def test_txt_chunks_have_no_header_path(tmp_path, config):
    txt_file = tmp_path / "docs" / "plain.txt"
    txt_file.write_text("Paragraph 1\n\nParagraph 2\n\nParagraph 3")

    parser = PlainTextParser()
    doc = parser.parse(str(txt_file))

    chunker = HeaderBasedChunker(config.chunking)
    chunks = chunker.chunk_record(_to_record(doc))

    for chunk in chunks:
        assert chunk.metadata.get("header_path") == ""


def test_txt_small_content_single_chunk(tmp_path, config):
    txt_file = tmp_path / "docs" / "small.txt"
    txt_file.write_text("This is a small text file with minimal content.")

    parser = PlainTextParser()
    doc = parser.parse(str(txt_file))

    chunker = HeaderBasedChunker(config.chunking)
    chunks = chunker.chunk_record(_to_record(doc))

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

    results = _search_records(manager, "SQL relational database")

    assert len(results) > 0
    found = False
    for result in results:
        if result is not None and "sql" in result.body.lower():
            found = True
            assert result.metadata["header_path"] == ""
            assert "database" in str(result.metadata["file_path"]).lower()
            break
    assert found


def test_mixed_md_and_txt_indexing(tmp_path, manager):
    md_file = tmp_path / "docs" / "readme.md"
    md_file.write_text("# Project\n\nMarkdown content here.")

    txt_file = tmp_path / "docs" / "notes.txt"
    txt_file.write_text("Plain text content here.")

    manager.index_document(str(md_file))
    manager.index_document(str(txt_file))

    md_results = _search_records(manager, "markdown")
    assert any("markdown" in r.body.lower() for r in md_results if r is not None)

    txt_results = _search_records(manager, "plain text")
    assert any(
        "plain text" in r.body.lower() for r in txt_results if r is not None
    )


def test_txt_chunk_start_end_positions(tmp_path, config):
    txt_file = tmp_path / "docs" / "positions.txt"
    content = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
    txt_file.write_text(content)

    parser = PlainTextParser()
    doc = parser.parse(str(txt_file))

    chunker = HeaderBasedChunker(config.chunking)
    chunks = chunker.chunk_record(_to_record(doc))

    for chunk in chunks:
        start_pos = int(chunk.metadata["start_pos"])
        end_pos = int(chunk.metadata["end_pos"])
        assert start_pos >= 0
        assert end_pos <= len(content)
        assert start_pos < end_pos


def test_txt_multiple_paragraphs_chunking(tmp_path, config):
    txt_file = tmp_path / "docs" / "multi.txt"
    paragraphs = [f"Paragraph {i} with some content." for i in range(10)]
    txt_file.write_text("\n\n".join(paragraphs))

    parser = PlainTextParser()
    doc = parser.parse(str(txt_file))

    chunker = HeaderBasedChunker(config.chunking)
    chunks = chunker.chunk_record(_to_record(doc))

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

    results = _search_records(manager, "international characters")
    assert len(results) > 0


def test_txt_empty_paragraphs_handled(tmp_path, config):
    txt_file = tmp_path / "docs" / "empty.txt"
    txt_file.write_text("Para 1\n\n\n\nPara 2\n\n\n\n\n\nPara 3")

    parser = PlainTextParser()
    doc = parser.parse(str(txt_file))

    chunker = HeaderBasedChunker(config.chunking)
    chunks = chunker.chunk_record(_to_record(doc))

    assert len(chunks) >= 1
    for chunk in chunks:
        assert chunk.content.strip()


def test_txt_metadata_preserved(tmp_path, config):
    txt_file = tmp_path / "docs" / "meta.txt"
    txt_file.write_text("Content with metadata.")

    parser = PlainTextParser()
    doc = parser.parse(str(txt_file))

    chunker = HeaderBasedChunker(config.chunking)
    chunks = chunker.chunk_record(_to_record(doc))

    for chunk in chunks:
        assert chunk.metadata.get("file_path") == str(txt_file)
        assert chunk.metadata.get("modified_time") == doc.modified_time.isoformat()
        assert chunk.record_id == doc.id


def test_txt_chunk_ids_unique(tmp_path, config):
    txt_file = tmp_path / "docs" / "test.txt"
    content = "\n\n".join([f"Paragraph {i}" * 100 for i in range(10)])
    txt_file.write_text(content)

    parser = PlainTextParser()
    doc = parser.parse(str(txt_file))

    chunker = HeaderBasedChunker(config.chunking)
    chunks = chunker.chunk_record(_to_record(doc))

    chunk_ids = [c.chunk_id for c in chunks]
    assert len(chunk_ids) == len(set(chunk_ids))
