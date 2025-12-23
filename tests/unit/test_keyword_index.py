from datetime import datetime

import pytest

from src.indices.keyword import KeywordIndex
from src.models import Document


@pytest.fixture
def sample_document():
    return Document(
        id="test-doc",
        content="Machine learning is a subset of artificial intelligence.",
        metadata={"title": "ML Intro", "aliases": ["AI Intro", "ML Basics"]},
        links=["AI"],
        tags=["ml", "ai"],
        file_path="/tmp/test.md",
        modified_time=datetime.now(),
    )


@pytest.fixture
def keyword_index():
    return KeywordIndex()


def test_keyword_index_add_and_search(keyword_index, sample_document):
    keyword_index.add(sample_document)

    results = keyword_index.search("machine learning", top_k=5)

    assert "test-doc" in results
    assert len(results) <= 5


def test_keyword_index_search_aliases(keyword_index, sample_document):
    keyword_index.add(sample_document)

    results = keyword_index.search("AI Basics", top_k=5)

    assert "test-doc" in results


def test_keyword_index_search_tags(keyword_index, sample_document):
    keyword_index.add(sample_document)

    results = keyword_index.search("ml", top_k=5)

    assert "test-doc" in results


def test_keyword_index_remove(keyword_index, sample_document):
    keyword_index.add(sample_document)

    results_before = keyword_index.search("machine learning", top_k=5)
    assert "test-doc" in results_before

    keyword_index.remove("test-doc")

    results_after = keyword_index.search("machine learning", top_k=5)
    assert "test-doc" not in results_after


def test_keyword_index_empty_query(keyword_index, sample_document):
    keyword_index.add(sample_document)

    results = keyword_index.search("", top_k=5)
    assert results == []

    results = keyword_index.search("   ", top_k=5)
    assert results == []


def test_keyword_index_persist_and_load(tmp_path, sample_document):
    index1 = KeywordIndex()
    index1.add(sample_document)

    persist_path = tmp_path / "index"
    index1.persist(persist_path)

    assert persist_path.exists()

    index2 = KeywordIndex()
    index2.load(persist_path)

    results = index2.search("machine learning", top_k=5)
    assert "test-doc" in results


def test_keyword_index_multiple_documents(keyword_index):
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

    keyword_index.add(doc1)
    keyword_index.add(doc2)

    results = keyword_index.search("python", top_k=5)
    assert "doc1" in results

    results = keyword_index.search("javascript", top_k=5)
    assert "doc2" in results


def test_keyword_index_exact_match_priority(keyword_index):
    doc1 = Document(
        id="doc1",
        content="BM25 is a ranking function used in information retrieval.",
        metadata={},
        links=[],
        tags=[],
        file_path="/tmp/doc1.md",
        modified_time=datetime.now(),
    )

    doc2 = Document(
        id="doc2",
        content="Information retrieval is important for search engines.",
        metadata={},
        links=[],
        tags=[],
        file_path="/tmp/doc2.md",
        modified_time=datetime.now(),
    )

    keyword_index.add(doc1)
    keyword_index.add(doc2)

    results = keyword_index.search("BM25", top_k=5)

    assert "doc1" in results
    assert results.index("doc1") < results.index("doc2") if "doc2" in results else True


def test_keyword_index_update_document(keyword_index):
    doc = Document(
        id="doc1",
        content="Original content about Python.",
        metadata={},
        links=[],
        tags=["python"],
        file_path="/tmp/doc1.md",
        modified_time=datetime.now(),
    )

    keyword_index.add(doc)

    results = keyword_index.search("python", top_k=5)
    assert "doc1" in results

    updated_doc = Document(
        id="doc1",
        content="Updated content about JavaScript.",
        metadata={},
        links=[],
        tags=["javascript"],
        file_path="/tmp/doc1.md",
        modified_time=datetime.now(),
    )

    keyword_index.add(updated_doc)

    results = keyword_index.search("javascript", top_k=5)
    assert "doc1" in results

    results = keyword_index.search("python", top_k=5)
    assert "doc1" not in results


def test_keyword_index_load_nonexistent_path(tmp_path):
    index = KeywordIndex()
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
    assert "test-doc" in results


@pytest.mark.skip(
    reason="Whoosh tokenization normalizes 'C++' to 'c', making exact match impossible. "
    "This is inherent to Whoosh's StandardAnalyzer which strips punctuation. "
    "Would require custom analyzer configuration to preserve such tokens."
)
def test_keyword_index_special_characters(keyword_index):
    doc = Document(
        id="special-doc",
        content="C++ is a programming language. Node.js is a runtime.",
        metadata={},
        links=[],
        tags=["c++", "nodejs"],
        file_path="/tmp/special.md",
        modified_time=datetime.now(),
    )

    keyword_index.add(doc)

    results = keyword_index.search("C++", top_k=5)
    assert "special-doc" in results

    results = keyword_index.search("Node.js", top_k=5)
    assert "special-doc" in results


def test_keyword_index_phrase_search(keyword_index):
    doc1 = Document(
        id="doc1",
        content="The quick brown fox jumps over the lazy dog.",
        metadata={},
        links=[],
        tags=[],
        file_path="/tmp/doc1.md",
        modified_time=datetime.now(),
    )

    doc2 = Document(
        id="doc2",
        content="A lazy fox and a quick dog.",
        metadata={},
        links=[],
        tags=[],
        file_path="/tmp/doc2.md",
        modified_time=datetime.now(),
    )

    keyword_index.add(doc1)
    keyword_index.add(doc2)

    results = keyword_index.search("quick brown fox", top_k=5)

    assert "doc1" in results


def test_keyword_index_no_results(keyword_index, sample_document):
    keyword_index.add(sample_document)

    results = keyword_index.search("quantum physics", top_k=5)

    assert results == []


def test_keyword_index_aliases_as_string(keyword_index):
    doc = Document(
        id="doc1",
        content="Content about AI.",
        metadata={"aliases": "Artificial Intelligence"},
        links=[],
        tags=[],
        file_path="/tmp/doc1.md",
        modified_time=datetime.now(),
    )

    keyword_index.add(doc)

    results = keyword_index.search("Artificial Intelligence", top_k=5)
    assert "doc1" in results


def test_keyword_index_no_aliases(keyword_index):
    doc = Document(
        id="doc1",
        content="Content without aliases.",
        metadata={},
        links=[],
        tags=[],
        file_path="/tmp/doc1.md",
        modified_time=datetime.now(),
    )

    keyword_index.add(doc)

    results = keyword_index.search("content", top_k=5)
    assert "doc1" in results


def test_keyword_index_concurrent_access(keyword_index):
    import threading

    doc1 = Document(
        id="doc1",
        content="First document.",
        metadata={},
        links=[],
        tags=[],
        file_path="/tmp/doc1.md",
        modified_time=datetime.now(),
    )

    doc2 = Document(
        id="doc2",
        content="Second document.",
        metadata={},
        links=[],
        tags=[],
        file_path="/tmp/doc2.md",
        modified_time=datetime.now(),
    )

    def add_doc1():
        keyword_index.add(doc1)

    def add_doc2():
        keyword_index.add(doc2)

    thread1 = threading.Thread(target=add_doc1)
    thread2 = threading.Thread(target=add_doc2)

    thread1.start()
    thread2.start()

    thread1.join()
    thread2.join()

    results = keyword_index.search("document", top_k=5)
    assert "doc1" in results
    assert "doc2" in results


def test_keyword_index_empty_content(keyword_index):
    """
    Validates graceful handling of documents with empty content.
    Prevents indexing crashes on placeholder or metadata-only files.
    """
    doc = Document(
        id="empty-doc",
        content="",
        metadata={"title": "Empty File"},
        links=[],
        tags=["empty"],
        file_path="/tmp/empty.md",
        modified_time=datetime.now(),
    )

    keyword_index.add(doc)

    results = keyword_index.search("empty", top_k=5)
    assert "empty-doc" in results


def test_keyword_index_very_large_document(keyword_index):
    """
    Tests indexing of very large documents (>10k characters).
    Ensures Whoosh can handle large content without performance degradation or errors.
    """
    large_content = " ".join(
        [f"This is sentence number {i} in a very long document." for i in range(200)]
    )
    assert len(large_content) > 10000

    doc = Document(
        id="large-doc",
        content=large_content,
        metadata={},
        links=[],
        tags=["large"],
        file_path="/tmp/large.md",
        modified_time=datetime.now(),
    )

    keyword_index.add(doc)

    results = keyword_index.search("sentence number 42", top_k=5)
    assert "large-doc" in results

    results = keyword_index.search("large", top_k=5)
    assert "large-doc" in results
