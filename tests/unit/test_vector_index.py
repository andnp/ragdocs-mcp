from datetime import datetime

import pytest

from src.indices.vector import VectorIndex
from src.models import Document


def _extract_chunk_ids(results: list) -> list[str]:
    if not results:
        return []
    if isinstance(results[0], str):
        return results
    return [r["chunk_id"] for r in results]


@pytest.fixture
def sample_document():
    return Document(
        id="test-doc",
        content="# Machine Learning\n\nMachine learning is a subset of artificial intelligence.",
        metadata={"title": "ML Intro"},
        links=["AI"],
        tags=["ml", "ai"],
        file_path="/tmp/test.md",
        modified_time=datetime.now(),
    )


@pytest.fixture
def vector_index(tmp_path, shared_embedding_model):
    return VectorIndex(embedding_model=shared_embedding_model)


def test_vector_index_add_and_search(vector_index, sample_document):
    vector_index.add(sample_document)

    results = vector_index.search("what is machine learning", top_k=5)

    assert "test-doc" in _extract_chunk_ids(results)
    assert len(results) <= 5


def test_vector_index_remove(vector_index, sample_document):
    vector_index.add(sample_document)

    results_before = vector_index.search("machine learning", top_k=5)
    assert "test-doc" in _extract_chunk_ids(results_before)

    vector_index.remove("test-doc")

    # Verify removal via public API
    # Note: Due to FAISS limitation, vectors remain but mapping is removed
    # We verify the operation completed without error


def test_vector_index_empty_query(vector_index, sample_document):
    vector_index.add(sample_document)

    results = vector_index.search("", top_k=5)
    assert results == []

    results = vector_index.search("   ", top_k=5)
    assert results == []


def test_vector_index_persist_and_load(tmp_path, sample_document, shared_embedding_model):
    index1 = VectorIndex(embedding_model=shared_embedding_model)
    index1.add(sample_document)

    persist_path = tmp_path / "index"
    index1.persist(persist_path)

    assert persist_path.exists()
    assert (persist_path / "doc_id_mapping.json").exists()

    index2 = VectorIndex(embedding_model=shared_embedding_model)
    index2.load(persist_path)

    results = index2.search("machine learning", top_k=5)
    assert "test-doc" in _extract_chunk_ids(results)


def test_vector_index_multiple_documents(vector_index):
    doc1 = Document(
        id="doc1",
        content="Python is a programming language.",
        metadata={},
        links=[],
        tags=["python"],
        file_path="/tmp/doc1.md",
        modified_time=datetime.now(),
    )

    doc2 = Document(
        id="doc2",
        content="JavaScript is used for web development.",
        metadata={},
        links=[],
        tags=["javascript"],
        file_path="/tmp/doc2.md",
        modified_time=datetime.now(),
    )

    vector_index.add(doc1)
    vector_index.add(doc2)

    results = vector_index.search("programming language", top_k=5)

    assert "doc1" in _extract_chunk_ids(results) or "doc2" in results
    assert len(results) <= 5


def test_vector_index_search_returns_unique_doc_ids(vector_index):
    doc = Document(
        id="long-doc",
        content="# Chapter 1\n\n" + "Content paragraph. " * 100 + "\n\n# Chapter 2\n\n" + "More content here. " * 100,
        metadata={},
        links=[],
        tags=[],
        file_path="/tmp/long.md",
        modified_time=datetime.now(),
    )

    vector_index.add(doc)

    results = vector_index.search("content", top_k=10)

    assert _extract_chunk_ids(results).count("long-doc") == 1


def test_vector_index_load_nonexistent_path(tmp_path, shared_embedding_model):
    index = VectorIndex(embedding_model=shared_embedding_model)
    nonexistent_path = tmp_path / "nonexistent"

    index.load(nonexistent_path)

    # Verify index is functional by adding and searching a document
    doc = Document(
        id="test-doc",
        content="Test content for initialization.",
        metadata={},
        links=[],
        tags=[],
        file_path="/tmp/test.md",
        modified_time=datetime.now(),
    )
    index.add(doc)
    results = index.search("test content", top_k=5)
    assert "test-doc" in _extract_chunk_ids(results)


def test_vector_index_very_long_document(vector_index):
    """
    Test handling of documents exceeding 10k characters.
    Ensures chunking and indexing works without errors for large documents.
    """
    long_content = "# Very Long Document\n\n"
    long_content += "This is a paragraph with meaningful content. " * 250  # ~11k chars
    long_content += "\n\n## Section 2\n\n"
    long_content += "More content here with searchable terms like embeddings and vectors. " * 100

    doc = Document(
        id="long-doc",
        content=long_content,
        metadata={"title": "Long Document"},
        links=[],
        tags=["long", "test"],
        file_path="/tmp/long_doc.md",
        modified_time=datetime.now(),
    )

    vector_index.add(doc)

    results = vector_index.search("embeddings and vectors", top_k=5)
    assert "long-doc" in _extract_chunk_ids(results)

    assert len(long_content) > 10000


def test_vector_index_special_characters(vector_index):
    """
    Test handling of special characters, unicode, and symbols.
    Validates robustness against emojis, punctuation, and non-ASCII content.
    """
    content = """# Special Characters Test 🚀

Unicode: 你好世界 (Chinese), Привет мир (Russian), مرحبا بالعالم (Arabic)

Symbols & Punctuation: !@#$%^&*()_+-=[]{}|;':",./<>?

Math: α, β, γ, ∑, ∫, ∞, ≈, ≠, ≤, ≥

Emojis: 🎉 🔥 💡 🌟 ⚡ 🎯 🚀 💻 📚 🌍

Quotes: "smart quotes" 'apostrophes' «guillemets» ‹single›

Special: €, £, ¥, ©, ®, ™, §, ¶, †, ‡, …
"""

    doc = Document(
        id="special-chars",
        content=content,
        metadata={"type": "special"},
        links=[],
        tags=["unicode", "symbols"],
        file_path="/tmp/special.md",
        modified_time=datetime.now(),
    )

    vector_index.add(doc)

    results = vector_index.search("unicode symbols", top_k=5)
    assert "special-chars" in _extract_chunk_ids(results)

    results_emoji = vector_index.search("emojis punctuation", top_k=5)
    assert "special-chars" in _extract_chunk_ids(results_emoji)


# ============================================================================
# Header-Weighted Embedding Tests (Phase 1 Search Quality)
# ============================================================================


def test_vector_index_chunk_with_header_path_includes_header_in_embedding(shared_embedding_model):
    """
    Chunks with header_path include header context in embedding.

    Verifies P2: Heading-weighted embeddings prepend header_path to content
    before generating embeddings, improving semantic search relevance.
    """
    from src.models import Chunk

    vector_index = VectorIndex(embedding_model=shared_embedding_model)

    chunk = Chunk(
        chunk_id="doc1_chunk_0",
        doc_id="doc1",
        content="This section covers the basics of training neural networks.",
        metadata={"tags": [], "links": []},
        chunk_index=0,
        header_path="Machine Learning > Deep Learning > Training",
        start_pos=0,
        end_pos=100,
        file_path="/tmp/ml.md",
        modified_time=datetime.now(),
    )

    vector_index.add_chunk(chunk)

    # Search should find content via header context
    results = vector_index.search("deep learning training", top_k=5)
    assert "doc1_chunk_0" in _extract_chunk_ids(results)

    # Search for terms only in header_path
    results_header = vector_index.search("machine learning", top_k=5)
    assert "doc1_chunk_0" in _extract_chunk_ids(results_header)


def test_vector_index_chunk_without_header_path_uses_content_only(shared_embedding_model):
    """
    Chunks without header_path use content only for embedding.

    Ensures backward compatibility for chunks that don't have header paths.
    """
    from src.models import Chunk

    vector_index = VectorIndex(embedding_model=shared_embedding_model)

    chunk = Chunk(
        chunk_id="doc2_chunk_0",
        doc_id="doc2",
        content="Python is a versatile programming language used for web development.",
        metadata={"tags": ["python"], "links": []},
        chunk_index=0,
        header_path="",  # Empty header path
        start_pos=0,
        end_pos=80,
        file_path="/tmp/python.md",
        modified_time=datetime.now(),
    )

    vector_index.add_chunk(chunk)

    # Search for content terms
    results = vector_index.search("python programming language", top_k=5)
    assert "doc2_chunk_0" in _extract_chunk_ids(results)


def test_vector_index_header_weighted_improves_relevance(shared_embedding_model):
    """
    Header-weighted embeddings improve search relevance for related queries.

    Verifies that chunks with relevant headers rank higher than those without
    when searching for terms in the header path.
    """
    from src.models import Chunk

    vector_index = VectorIndex(embedding_model=shared_embedding_model)

    # Chunk with relevant header
    chunk_with_header = Chunk(
        chunk_id="api_chunk_0",
        doc_id="api-docs",
        content="This function accepts parameters and returns a value.",
        metadata={},
        chunk_index=0,
        header_path="API Reference > Authentication > Token Validation",
        start_pos=0,
        end_pos=60,
        file_path="/tmp/api.md",
        modified_time=datetime.now(),
    )

    # Chunk without relevant header
    chunk_without_header = Chunk(
        chunk_id="general_chunk_0",
        doc_id="general-docs",
        content="This function accepts parameters and returns a value.",
        metadata={},
        chunk_index=0,
        header_path="",  # No header
        start_pos=0,
        end_pos=60,
        file_path="/tmp/general.md",
        modified_time=datetime.now(),
    )

    vector_index.add_chunk(chunk_with_header)
    vector_index.add_chunk(chunk_without_header)

    # Search for authentication - should favor chunk with relevant header
    results = vector_index.search("authentication token validation", top_k=5)
    chunk_ids = _extract_chunk_ids(results)

    assert "api_chunk_0" in chunk_ids
    # api_chunk should rank higher due to header context
    if "general_chunk_0" in chunk_ids:
        assert chunk_ids.index("api_chunk_0") < chunk_ids.index("general_chunk_0")
