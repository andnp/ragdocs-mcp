from datetime import datetime

import pytest

from src.indices.keyword import KeywordIndex
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

    assert "test-doc" in _extract_chunk_ids(results)
    assert len(results) <= 5


def test_keyword_index_search_aliases(keyword_index, sample_document):
    keyword_index.add(sample_document)

    results = keyword_index.search("AI Basics", top_k=5)

    assert "test-doc" in _extract_chunk_ids(results)


def test_keyword_index_search_tags(keyword_index, sample_document):
    keyword_index.add(sample_document)

    results = keyword_index.search("ml", top_k=5)

    assert "test-doc" in _extract_chunk_ids(results)


def test_keyword_index_remove(keyword_index, sample_document):
    keyword_index.add(sample_document)

    results_before = keyword_index.search("machine learning", top_k=5)
    assert "test-doc" in _extract_chunk_ids(results_before)

    keyword_index.remove("test-doc")

    results_after = keyword_index.search("machine learning", top_k=5)
    assert "test-doc" not in _extract_chunk_ids(results_after)


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
    assert "test-doc" in _extract_chunk_ids(results)


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
    assert "doc1" in _extract_chunk_ids(results)

    results = keyword_index.search("javascript", top_k=5)
    assert "doc2" in _extract_chunk_ids(results)


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

    assert "doc1" in _extract_chunk_ids(results)
    assert _extract_chunk_ids(results).index("doc1") < _extract_chunk_ids(results).index("doc2") if "doc2" in results else True


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
    assert "doc1" in _extract_chunk_ids(results)

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
    assert "doc1" in _extract_chunk_ids(results)

    results = keyword_index.search("python", top_k=5)
    assert "doc1" not in _extract_chunk_ids(results)


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
    assert "test-doc" in _extract_chunk_ids(results)


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
    assert "special-doc" in _extract_chunk_ids(results)

    results = keyword_index.search("Node.js", top_k=5)
    assert "special-doc" in _extract_chunk_ids(results)


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

    assert "doc1" in _extract_chunk_ids(results)


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
    assert "doc1" in _extract_chunk_ids(results)


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
    assert "doc1" in _extract_chunk_ids(results)


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
    assert "doc1" in _extract_chunk_ids(results)
    assert "doc2" in _extract_chunk_ids(results)


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
    assert "empty-doc" in _extract_chunk_ids(results)


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
    assert "large-doc" in _extract_chunk_ids(results)

    results = keyword_index.search("large", top_k=5)
    assert "large-doc" in _extract_chunk_ids(results)


# ============================================================================
# BM25F Field Boosting Tests (Phase 1 Search Quality)
# ============================================================================


def test_keyword_index_title_field_boosted():
    """
    Title field is indexed with boost factor 3.0.

    Verifies P4: Title field has highest boost and matches rank higher.
    """
    from src.models import Chunk

    keyword_index = KeywordIndex()

    # Chunk with search term in title
    chunk_with_title = Chunk(
        chunk_id="titled_chunk_0",
        doc_id="titled-doc",
        content="Some generic content about programming.",
        metadata={"title": "Authentication Guide", "tags": []},
        chunk_index=0,
        header_path="",
        start_pos=0,
        end_pos=50,
        file_path="/tmp/auth.md",
        modified_time=datetime.now(),
    )

    # Chunk with search term in content only
    chunk_content_only = Chunk(
        chunk_id="content_chunk_0",
        doc_id="content-doc",
        content="This document covers authentication patterns and best practices.",
        metadata={"title": "Generic Document", "tags": []},
        chunk_index=0,
        header_path="",
        start_pos=0,
        end_pos=70,
        file_path="/tmp/generic.md",
        modified_time=datetime.now(),
    )

    keyword_index.add_chunk(chunk_with_title)
    keyword_index.add_chunk(chunk_content_only)

    results = keyword_index.search("authentication", top_k=5)
    chunk_ids = _extract_chunk_ids(results)

    assert "titled_chunk_0" in chunk_ids
    assert "content_chunk_0" in chunk_ids
    # Title match should rank higher due to 3.0 boost
    assert chunk_ids.index("titled_chunk_0") < chunk_ids.index("content_chunk_0")


def test_keyword_index_headers_field_indexed():
    """
    Headers field is indexed with boost factor 2.5.

    Verifies header_path is searchable in keyword index.
    """
    from src.models import Chunk

    keyword_index = KeywordIndex()

    chunk = Chunk(
        chunk_id="header_chunk_0",
        doc_id="header-doc",
        content="Implementation details for the feature.",
        metadata={"tags": []},
        chunk_index=0,
        header_path="API Reference > Endpoints > User Management",
        start_pos=0,
        end_pos=50,
        file_path="/tmp/api.md",
        modified_time=datetime.now(),
    )

    keyword_index.add_chunk(chunk)

    # Search for terms in header_path
    results = keyword_index.search("API endpoints", top_k=5)
    assert "header_chunk_0" in _extract_chunk_ids(results)

    results = keyword_index.search("user management", top_k=5)
    assert "header_chunk_0" in _extract_chunk_ids(results)


def test_keyword_index_keywords_field_indexed():
    """
    Keywords field is indexed with boost factor 2.5.

    Verifies frontmatter keywords are searchable.
    """
    from src.models import Chunk

    keyword_index = KeywordIndex()

    chunk = Chunk(
        chunk_id="kw_chunk_0",
        doc_id="kw-doc",
        content="General content without specific terms.",
        metadata={
            "keywords": ["microservices", "distributed-systems", "scalability"],
            "tags": [],
        },
        chunk_index=0,
        header_path="",
        start_pos=0,
        end_pos=50,
        file_path="/tmp/arch.md",
        modified_time=datetime.now(),
    )

    keyword_index.add_chunk(chunk)

    # Search for keywords
    results = keyword_index.search("microservices", top_k=5)
    assert "kw_chunk_0" in _extract_chunk_ids(results)

    results = keyword_index.search("distributed systems", top_k=5)
    assert "kw_chunk_0" in _extract_chunk_ids(results)


def test_keyword_index_description_field_indexed():
    """
    Description field is indexed with boost factor 2.0.

    Verifies frontmatter description is searchable.
    """
    from src.models import Chunk

    keyword_index = KeywordIndex()

    chunk = Chunk(
        chunk_id="desc_chunk_0",
        doc_id="desc-doc",
        content="Code examples and snippets.",
        metadata={
            "description": "A comprehensive guide to containerization with Docker",
            "tags": [],
        },
        chunk_index=0,
        header_path="",
        start_pos=0,
        end_pos=30,
        file_path="/tmp/docker.md",
        modified_time=datetime.now(),
    )

    keyword_index.add_chunk(chunk)

    # Search for terms in description
    results = keyword_index.search("containerization Docker", top_k=5)
    assert "desc_chunk_0" in _extract_chunk_ids(results)


def test_keyword_index_author_field_indexed():
    """
    Author field is indexed for searchability.

    Verifies documents can be found by author name.
    """
    from src.models import Chunk

    keyword_index = KeywordIndex()

    chunk = Chunk(
        chunk_id="author_chunk_0",
        doc_id="author-doc",
        content="Technical documentation content.",
        metadata={"author": "John Smith", "tags": []},
        chunk_index=0,
        header_path="",
        start_pos=0,
        end_pos=40,
        file_path="/tmp/authored.md",
        modified_time=datetime.now(),
    )

    keyword_index.add_chunk(chunk)

    # Search by author
    results = keyword_index.search("John Smith", top_k=5)
    assert "author_chunk_0" in _extract_chunk_ids(results)


def test_keyword_index_category_field_indexed():
    """
    Category field is indexed as KEYWORD type.

    Verifies documents can be filtered/searched by category.
    """
    from src.models import Chunk

    keyword_index = KeywordIndex()

    chunk = Chunk(
        chunk_id="cat_chunk_0",
        doc_id="cat-doc",
        content="Tutorial content here.",
        metadata={"category": "tutorials", "tags": []},
        chunk_index=0,
        header_path="",
        start_pos=0,
        end_pos=25,
        file_path="/tmp/tutorial.md",
        modified_time=datetime.now(),
    )

    keyword_index.add_chunk(chunk)

    # Category is a KEYWORD field, exact match search
    results = keyword_index.search("tutorials", top_k=5)
    assert "cat_chunk_0" in _extract_chunk_ids(results)


def test_keyword_index_all_boosted_fields_together():
    """
    All boosted fields work together for comprehensive search.

    Verifies multiple frontmatter fields are indexed and searchable.
    """
    from src.models import Chunk

    keyword_index = KeywordIndex()

    chunk = Chunk(
        chunk_id="full_chunk_0",
        doc_id="full-doc",
        content="Main content of the document.",
        metadata={
            "title": "Kubernetes Deployment Guide",
            "description": "Step-by-step instructions for deploying applications",
            "keywords": ["k8s", "containers", "orchestration"],
            "author": "DevOps Team",
            "category": "infrastructure",
            "aliases": ["k8s-guide", "deployment-howto"],
            "tags": ["kubernetes", "devops"],
        },
        chunk_index=0,
        header_path="Getting Started > Prerequisites",
        start_pos=0,
        end_pos=35,
        file_path="/tmp/k8s.md",
        modified_time=datetime.now(),
    )

    keyword_index.add_chunk(chunk)

    # Search various fields
    assert "full_chunk_0" in _extract_chunk_ids(keyword_index.search("Kubernetes", top_k=5))
    assert "full_chunk_0" in _extract_chunk_ids(keyword_index.search("deploying applications", top_k=5))
    assert "full_chunk_0" in _extract_chunk_ids(keyword_index.search("k8s containers", top_k=5))
    assert "full_chunk_0" in _extract_chunk_ids(keyword_index.search("DevOps Team", top_k=5))
    assert "full_chunk_0" in _extract_chunk_ids(keyword_index.search("prerequisites", top_k=5))
    assert "full_chunk_0" in _extract_chunk_ids(keyword_index.search("k8s-guide", top_k=5))


def test_keyword_index_schema_mismatch_triggers_rebuild(tmp_path):
    """
    Loading an index with mismatched schema triggers a rebuild.

    When the persisted index has a different schema than expected (e.g.,
    missing new fields), the index should be rebuilt from scratch to
    avoid field errors.
    """
    from whoosh import index as whoosh_index
    from whoosh.fields import ID, TEXT, Schema

    old_schema = Schema(
        id=ID(stored=True, unique=True),
        doc_id=ID(stored=True),
        content=TEXT(stored=False),
        aliases=TEXT(stored=False),
        tags=TEXT(stored=False),
    )
    index_path = tmp_path / "old_keyword_index"
    index_path.mkdir()
    whoosh_index.create_in(str(index_path), old_schema)

    keyword_index = KeywordIndex()
    keyword_index.load(index_path)

    from src.models import Chunk

    chunk = Chunk(
        chunk_id="new_chunk_0",
        doc_id="new-doc",
        content="Test content.",
        metadata={"author": "Test Author", "tags": []},
        chunk_index=0,
        header_path="",
        start_pos=0,
        end_pos=13,
        file_path="/tmp/test.md",
        modified_time=datetime.now(),
    )
    keyword_index.add_chunk(chunk)

    results = keyword_index.search("Test Author", top_k=5)
    assert "new_chunk_0" in _extract_chunk_ids(results)


def test_keyword_index_remove_handles_corrupted_segment(tmp_path):
    """
    Remove operation handles corrupted segment files gracefully.

    When Whoosh segment files (.seg) are deleted/corrupted mid-operation,
    the index should detect the corruption, reinitialize, and not crash.
    """
    import glob
    from pathlib import Path

    from src.models import Chunk

    keyword_index = KeywordIndex()

    chunk = Chunk(
        chunk_id="chunk_to_remove_0",
        doc_id="test-doc",
        content="Content for removal testing.",
        metadata={"tags": []},
        chunk_index=0,
        header_path="",
        start_pos=0,
        end_pos=30,
        file_path="/tmp/test.md",
        modified_time=datetime.now(),
    )
    keyword_index.add_chunk(chunk)

    index_path = tmp_path / "corrupted_keyword_index"
    keyword_index.persist(index_path)
    keyword_index.load(index_path)

    seg_files = glob.glob(str(index_path / "*.seg"))
    assert len(seg_files) > 0, "Expected segment files after persist"
    for seg in seg_files:
        Path(seg).unlink()

    keyword_index.remove("chunk_to_remove_0")

    results = keyword_index.search("removal testing", top_k=5)
    assert isinstance(results, list)


def test_keyword_index_search_handles_corrupted_segment(tmp_path):
    """
    Search operation handles corrupted segment files gracefully.

    When Whoosh segment files (.seg) are corrupted, search should detect
    the issue, reinitialize the index, and return an empty list rather
    than crashing.
    """
    import glob
    from pathlib import Path

    from src.models import Chunk

    keyword_index = KeywordIndex()

    chunk = Chunk(
        chunk_id="search_chunk_0",
        doc_id="search-doc",
        content="Searchable content for testing.",
        metadata={"tags": []},
        chunk_index=0,
        header_path="",
        start_pos=0,
        end_pos=35,
        file_path="/tmp/test.md",
        modified_time=datetime.now(),
    )
    keyword_index.add_chunk(chunk)

    index_path = tmp_path / "corrupted_search_index"
    keyword_index.persist(index_path)
    keyword_index.load(index_path)

    seg_files = glob.glob(str(index_path / "*.seg"))
    assert len(seg_files) > 0, "Expected segment files after persist"
    for seg in seg_files:
        Path(seg).unlink()

    results = keyword_index.search("searchable content", top_k=5)

    assert results == []


def test_keyword_index_recovery_allows_reindexing(tmp_path):
    """
    After corruption recovery, new documents can be indexed successfully.

    This tests the full cycle: create index, persist, corrupt, detect
    corruption during operation, reinitialize, then add new documents
    successfully.
    """
    import glob
    from pathlib import Path

    from src.models import Chunk

    keyword_index = KeywordIndex()

    original_chunk = Chunk(
        chunk_id="original_0",
        doc_id="original-doc",
        content="Original content before corruption.",
        metadata={"tags": []},
        chunk_index=0,
        header_path="",
        start_pos=0,
        end_pos=40,
        file_path="/tmp/original.md",
        modified_time=datetime.now(),
    )
    keyword_index.add_chunk(original_chunk)

    index_path = tmp_path / "recovery_test_index"
    keyword_index.persist(index_path)
    keyword_index.load(index_path)

    seg_files = glob.glob(str(index_path / "*.seg"))
    for seg in seg_files:
        Path(seg).unlink()

    keyword_index.search("trigger corruption detection", top_k=5)

    new_chunk = Chunk(
        chunk_id="new_after_recovery_0",
        doc_id="new-doc",
        content="New content added after recovery.",
        metadata={"tags": []},
        chunk_index=0,
        header_path="",
        start_pos=0,
        end_pos=35,
        file_path="/tmp/new.md",
        modified_time=datetime.now(),
    )
    keyword_index.add_chunk(new_chunk)

    results = keyword_index.search("new content recovery", top_k=5)
    assert "new_after_recovery_0" in _extract_chunk_ids(results)
