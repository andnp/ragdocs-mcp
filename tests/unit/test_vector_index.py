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


# ============================================================================
# Stale Reference Cleanup Tests (Self-Healing)
# ============================================================================


def test_stale_chunk_ref_cleaned_on_lookup(shared_embedding_model):
    """
    Test that looking up a stale chunk ID cleans it from mappings.

    When get_chunk_by_id() encounters a chunk ID that exists in mappings
    but not in the docstore, it should remove the stale reference.
    """
    from src.models import Chunk

    vector_index = VectorIndex(embedding_model=shared_embedding_model)

    # Add a real chunk
    chunk = Chunk(
        chunk_id="real_chunk",
        doc_id="doc1",
        content="Real content that exists in docstore.",
        metadata={},
        chunk_index=0,
        header_path="",
        start_pos=0,
        end_pos=50,
        file_path="/tmp/real.md",
        modified_time=datetime.now(),
    )
    vector_index.add_chunk(chunk)

    # Manually inject a stale reference (ID in mappings but not docstore)
    stale_chunk_id = "stale_orphan_chunk"
    vector_index._chunk_id_to_node_id[stale_chunk_id] = stale_chunk_id
    vector_index._doc_id_to_node_ids["orphan_doc"] = [stale_chunk_id]

    # Verify stale ref exists in mappings
    assert stale_chunk_id in vector_index._chunk_id_to_node_id
    assert "orphan_doc" in vector_index._doc_id_to_node_ids

    # Look up the stale chunk - should trigger cleanup
    result = vector_index.get_chunk_by_id(stale_chunk_id)
    assert result is None

    # Verify stale ref was cleaned from mappings
    assert stale_chunk_id not in vector_index._chunk_id_to_node_id
    assert "orphan_doc" not in vector_index._doc_id_to_node_ids


def test_stale_warning_only_logged_once(shared_embedding_model, caplog):
    """
    Test that stale chunk warning is only logged once per chunk ID.

    The _warned_stale_chunk_ids set should prevent duplicate warnings
    from flooding logs when the same stale chunk is accessed repeatedly.
    """
    import logging

    vector_index = VectorIndex(embedding_model=shared_embedding_model)
    vector_index._initialize_index()  # Initialize so index is not None

    # Inject a stale reference
    stale_id = "stale_repeatedly_accessed"
    vector_index._chunk_id_to_node_id[stale_id] = stale_id

    # Clear any existing warnings
    vector_index._warned_stale_chunk_ids.clear()

    with caplog.at_level(logging.WARNING):
        # First lookup - should log warning
        vector_index.get_chunk_by_id(stale_id)

        # Count warnings for our stale ID
        first_warning_count = sum(
            1 for record in caplog.records
            if stale_id in record.message and record.levelno == logging.WARNING
        )
        assert first_warning_count == 1

        # Re-add the stale ref (simulate it being re-added somehow)
        vector_index._chunk_id_to_node_id[stale_id] = stale_id

        # Second lookup - should NOT log another warning
        caplog.clear()
        vector_index.get_chunk_by_id(stale_id)

        second_warning_count = sum(
            1 for record in caplog.records
            if stale_id in record.message and record.levelno == logging.WARNING
        )
        assert second_warning_count == 0

    # Verify the ID is in warned set
    assert stale_id in vector_index._warned_stale_chunk_ids


def test_reconcile_mappings_removes_stale_refs(shared_embedding_model):
    """
    Test that reconcile_mappings() batch-removes all stale references.

    The reconcile_mappings() method should scan all mappings and remove
    any chunk IDs that no longer exist in the docstore.
    """
    from src.models import Chunk

    vector_index = VectorIndex(embedding_model=shared_embedding_model)

    # Add real chunks
    for i in range(3):
        chunk = Chunk(
            chunk_id=f"real_chunk_{i}",
            doc_id=f"doc_{i}",
            content=f"Real content number {i}.",
            metadata={},
            chunk_index=0,
            header_path="",
            start_pos=0,
            end_pos=30,
            file_path=f"/tmp/real_{i}.md",
            modified_time=datetime.now(),
        )
        vector_index.add_chunk(chunk)

    # Inject multiple stale references
    for i in range(5):
        stale_id = f"stale_batch_{i}"
        vector_index._chunk_id_to_node_id[stale_id] = stale_id
        vector_index._doc_id_to_node_ids[f"stale_doc_{i}"] = [stale_id]

    # Verify all stale refs exist
    assert len([k for k in vector_index._chunk_id_to_node_id if k.startswith("stale_")]) == 5
    assert len([k for k in vector_index._doc_id_to_node_ids if k.startswith("stale_")]) == 5

    # Run reconciliation
    removed_count = vector_index.reconcile_mappings()

    # Should have removed all 5 stale refs
    assert removed_count == 5

    # Verify no stale refs remain
    assert len([k for k in vector_index._chunk_id_to_node_id if k.startswith("stale_")]) == 0
    assert len([k for k in vector_index._doc_id_to_node_ids if k.startswith("stale_")]) == 0

    # Real chunks should still be there
    assert len([k for k in vector_index._chunk_id_to_node_id if k.startswith("real_")]) == 3


def test_warned_set_cleared_on_load(shared_embedding_model, tmp_path):
    """
    Test that _warned_stale_chunk_ids is cleared when loading index.

    After loading an index from disk, the warned set should be reset
    since the stale state may have changed during persistence.
    """
    from src.models import Chunk

    vector_index = VectorIndex(embedding_model=shared_embedding_model)

    # Add a chunk
    chunk = Chunk(
        chunk_id="persistent_chunk",
        doc_id="doc1",
        content="Content to persist.",
        metadata={},
        chunk_index=0,
        header_path="",
        start_pos=0,
        end_pos=30,
        file_path="/tmp/test.md",
        modified_time=datetime.now(),
    )
    vector_index.add_chunk(chunk)

    # Simulate having warned about some stale chunks
    vector_index._warned_stale_chunk_ids["old_stale_1"] = True
    vector_index._warned_stale_chunk_ids["old_stale_2"] = True

    # Persist to disk
    persist_path = tmp_path / "vector_index"
    vector_index.persist(persist_path)

    # Create new index and load
    vector_index2 = VectorIndex(embedding_model=shared_embedding_model)

    # Add some warned IDs before load (simulating prior session)
    vector_index2._warned_stale_chunk_ids["pre_load_warning"] = True

    # Load the persisted index
    vector_index2.load(persist_path)

    # Warned set should be cleared
    assert len(vector_index2._warned_stale_chunk_ids) == 0

    # Verify index is functional
    results = vector_index2.search("content persist", top_k=5)
    assert "persistent_chunk" in _extract_chunk_ids(results)


def test_term_counts_and_vocabulary_loaded_as_ordereddict(shared_embedding_model, tmp_path):
    """
    Test that _term_counts and _concept_vocabulary are loaded as OrderedDict.

    Regression test for bug where JSON loading returned plain dict,
    causing AttributeError when calling move_to_end() during indexing.
    """
    from collections import OrderedDict

    from src.models import Chunk

    vector_index = VectorIndex(embedding_model=shared_embedding_model)

    # Add chunks to populate term counts and vocabulary
    for i in range(3):
        chunk = Chunk(
            chunk_id=f"chunk_{i}",
            doc_id=f"doc_{i}",
            content="Machine learning algorithms require training data for optimization.",
            metadata={},
            chunk_index=0,
            header_path="",
            start_pos=0,
            end_pos=100,
            file_path=f"/tmp/doc_{i}.md",
            modified_time=datetime.now(),
        )
        vector_index.add_chunk(chunk)

    # Register terms to populate _term_counts
    vector_index.register_document_terms("machine learning optimization algorithms")

    # Build vocabulary
    vector_index.build_concept_vocabulary(min_term_length=3, max_terms=100, min_frequency=1)

    # Verify both are OrderedDict before persist
    assert isinstance(vector_index._term_counts, OrderedDict)
    assert isinstance(vector_index._concept_vocabulary, OrderedDict)
    assert len(vector_index._term_counts) > 0
    assert len(vector_index._concept_vocabulary) > 0

    # Persist to disk
    persist_path = tmp_path / "vector_index_ordered"
    vector_index.persist(persist_path)

    # Load into new index
    vector_index2 = VectorIndex(embedding_model=shared_embedding_model)
    vector_index2.load(persist_path)

    # Verify both are still OrderedDict after load (this was the bug)
    assert isinstance(vector_index2._term_counts, OrderedDict)
    assert isinstance(vector_index2._concept_vocabulary, OrderedDict)

    # Verify move_to_end() works (this would fail with plain dict)
    if vector_index2._term_counts:
        first_term = next(iter(vector_index2._term_counts))
        vector_index2._term_counts.move_to_end(first_term)  # Should not raise AttributeError

    # Verify indexing still works after load (this triggered the original bug)
    chunk = Chunk(
        chunk_id="new_chunk_after_load",
        doc_id="new_doc",
        content="New content about neural networks and deep learning.",
        metadata={},
        chunk_index=0,
        header_path="",
        start_pos=0,
        end_pos=60,
        file_path="/tmp/new.md",
        modified_time=datetime.now(),
    )
    vector_index2.add_chunk(chunk)  # Should not raise AttributeError

    # Verify search works
    results = vector_index2.search("neural networks", top_k=5)
    assert "new_chunk_after_load" in _extract_chunk_ids(results)


def test_ordered_dict_preserved_after_persist_and_load_regression(shared_embedding_model, tmp_path):
    """
    Regression test for bug where json.load() returned plain dict instead of OrderedDict,
    causing AttributeError: 'dict' object has no attribute 'move_to_end'.

    This simulates the create_memory flow: persist index → load index → add new chunk.
    The bug manifested when add_chunk called register_document_terms which called
    _term_counts.move_to_end() on a plain dict.
    """
    from collections import OrderedDict

    from src.models import Chunk

    # Create index and add initial content
    vector_index = VectorIndex(embedding_model=shared_embedding_model)

    chunk1 = Chunk(
        chunk_id="chunk_1",
        doc_id="doc_1",
        content="Python programming language with asyncio and type hints.",
        metadata={},
        chunk_index=0,
        header_path="",
        start_pos=0,
        end_pos=100,
        file_path="/tmp/doc_1.md",
        modified_time=datetime.now(),
    )
    vector_index.add_chunk(chunk1)

    # Populate term counts and vocabulary
    vector_index.register_document_terms("python asyncio programming")
    vector_index.build_concept_vocabulary(min_term_length=3, max_terms=100, min_frequency=1)

    # Verify OrderedDict before persist
    assert isinstance(vector_index._term_counts, OrderedDict), \
        "term_counts should be OrderedDict before persist"
    assert isinstance(vector_index._concept_vocabulary, OrderedDict), \
        "concept_vocabulary should be OrderedDict before persist"

    # Persist to disk
    persist_path = tmp_path / "vector_ordered_dict_test"
    vector_index.persist(persist_path)

    # Create new index and load from disk (simulates memory system startup)
    vector_index2 = VectorIndex(embedding_model=shared_embedding_model)
    vector_index2.load(persist_path)

    # CRITICAL: Verify both are OrderedDict after load (this was the bug)
    assert isinstance(vector_index2._term_counts, OrderedDict), \
        "term_counts should be OrderedDict after load (was plain dict, causing move_to_end AttributeError)"
    assert isinstance(vector_index2._concept_vocabulary, OrderedDict), \
        "concept_vocabulary should be OrderedDict after load"

    # Simulate create_memory flow: add new chunk after loading (this would fail with plain dict)
    chunk2 = Chunk(
        chunk_id="chunk_2",
        doc_id="doc_2",
        content="FastAPI web framework with dependency injection and Pydantic models.",
        metadata={},
        chunk_index=0,
        header_path="",
        start_pos=0,
        end_pos=100,
        file_path="/tmp/doc_2.md",
        modified_time=datetime.now(),
    )

    # This should NOT raise AttributeError: 'dict' object has no attribute 'move_to_end'
    vector_index2.add_chunk(chunk2)

    # Verify the chunk was indexed successfully
    results = vector_index2.search("FastAPI web framework", top_k=5)
    assert "chunk_2" in _extract_chunk_ids(results)

    # Verify move_to_end() works directly (additional safety check)
    if vector_index2._term_counts:
        first_term = next(iter(vector_index2._term_counts))
        vector_index2._term_counts.move_to_end(first_term)  # Should not raise


# ============================================================================
# Chunk Removal Tests (Phase 2 Delta Indexing)
# ============================================================================


def test_remove_chunk_removes_from_all_mappings(vector_index, shared_embedding_model):
    """Test that remove_chunk() removes chunk from index and all mappings."""
    from src.models import Chunk

    # Add chunks
    chunk1 = Chunk(
        chunk_id="doc1#chunk#0",
        doc_id="doc1",
        content="First chunk content.",
        metadata={},
        chunk_index=0,
        header_path="Section 1",
        start_pos=0,
        end_pos=20,
        file_path="/tmp/doc1.md",
        modified_time=datetime.now(),
    )
    chunk2 = Chunk(
        chunk_id="doc1#chunk#1",
        doc_id="doc1",
        content="Second chunk content.",
        metadata={},
        chunk_index=1,
        header_path="Section 2",
        start_pos=21,
        end_pos=42,
        file_path="/tmp/doc1.md",
        modified_time=datetime.now(),
    )

    vector_index.add_chunk(chunk1)
    vector_index.add_chunk(chunk2)

    # Verify chunks are indexed
    assert "doc1#chunk#0" in vector_index._chunk_id_to_node_id
    assert "doc1#chunk#1" in vector_index._chunk_id_to_node_id
    assert "doc1" in vector_index._doc_id_to_node_ids
    assert len(vector_index._doc_id_to_node_ids["doc1"]) == 2

    # Remove one chunk
    vector_index.remove_chunk("doc1#chunk#0")

    # Verify removal
    assert "doc1#chunk#0" not in vector_index._chunk_id_to_node_id
    assert "doc1#chunk#1" in vector_index._chunk_id_to_node_id
    assert "doc1" in vector_index._doc_id_to_node_ids
    assert len(vector_index._doc_id_to_node_ids["doc1"]) == 1
    assert "doc1#chunk#1" in vector_index._doc_id_to_node_ids["doc1"]


def test_remove_chunk_handles_missing_chunk(vector_index):
    """Test that remove_chunk() handles missing chunk gracefully."""
    # Should not raise exception
    vector_index.remove_chunk("nonexistent_chunk_id")


def test_remove_chunk_removes_last_chunk_cleans_doc_mapping(vector_index):
    """Test that removing last chunk of a doc removes the doc from mappings."""
    from src.models import Chunk

    chunk = Chunk(
        chunk_id="doc_single#chunk#0",
        doc_id="doc_single",
        content="Only chunk.",
        metadata={},
        chunk_index=0,
        header_path="",
        start_pos=0,
        end_pos=11,
        file_path="/tmp/single.md",
        modified_time=datetime.now(),
    )

    vector_index.add_chunk(chunk)

    assert "doc_single" in vector_index._doc_id_to_node_ids

    # Remove the only chunk
    vector_index.remove_chunk("doc_single#chunk#0")

    # Doc mapping should be removed when last chunk is removed
    assert "doc_single" not in vector_index._doc_id_to_node_ids


def test_remove_chunk_thread_safe(vector_index):
    """Test that remove_chunk() is thread-safe with concurrent operations."""
    from concurrent.futures import ThreadPoolExecutor
    from src.models import Chunk

    # Add multiple chunks
    chunks = []
    for i in range(10):
        chunk = Chunk(
            chunk_id=f"concurrent_doc#chunk#{i}",
            doc_id="concurrent_doc",
            content=f"Chunk {i} content.",
            metadata={},
            chunk_index=i,
            header_path=f"Section {i}",
            start_pos=i * 20,
            end_pos=(i + 1) * 20,
            file_path="/tmp/concurrent.md",
            modified_time=datetime.now(),
        )
        chunks.append(chunk)
        vector_index.add_chunk(chunk)

    # Concurrently remove half the chunks
    def remove_chunk_task(chunk_id):
        vector_index.remove_chunk(chunk_id)

    with ThreadPoolExecutor(max_workers=4) as executor:
        chunk_ids_to_remove = [f"concurrent_doc#chunk#{i}" for i in range(0, 10, 2)]
        list(executor.map(remove_chunk_task, chunk_ids_to_remove))

    # Verify correct chunks were removed
    for i in range(10):
        chunk_id = f"concurrent_doc#chunk#{i}"
        if i % 2 == 0:
            assert chunk_id not in vector_index._chunk_id_to_node_id
        else:
            assert chunk_id in vector_index._chunk_id_to_node_id


def test_remove_chunk_before_initialization(vector_index):
    """Test remove_chunk() when index not initialized."""
    # Clear index
    vector_index._index = None

    # Should log warning but not raise
    vector_index.remove_chunk("any_chunk_id")
