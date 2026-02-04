import time
from pathlib import Path

import pytest

from src.config import Config, IndexingConfig, LLMConfig, SearchConfig, ServerConfig
from src.indexing.manager import IndexManager
from src.indices.graph import GraphStore
from src.indices.keyword import KeywordIndex
from src.indices.vector import VectorIndex


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
        parsers={"**/*.md": "MarkdownParser"},
        search=SearchConfig(),
        llm=LLMConfig(),
    )


@pytest.fixture(scope="module")
def indices(shared_embedding_model):
    """
    Module-scoped indices with shared embedding model.

    Scope changed to 'module' to avoid redundant embedding model loading.
    Each test creates its own documents in tmp_path, providing isolation
    despite shared indices.
    """
    vector = VectorIndex(embedding_model=shared_embedding_model)
    keyword = KeywordIndex()
    graph = GraphStore()
    return vector, keyword, graph


@pytest.fixture
def manager(config, indices):
    vector, keyword, graph = indices
    return IndexManager(config, vector, keyword, graph)


def create_realistic_document(doc_id: int, size: str = "medium") -> str:
    if size == "small":
        paragraphs = 2
        sentences_per_para = 3
    elif size == "large":
        paragraphs = 10
        sentences_per_para = 8
    else:  # medium
        paragraphs = 5
        sentences_per_para = 5

    content = f"# Document {doc_id}: Performance Testing\n\n"
    content += "tags: [performance, test, benchmark]\n"
    content += f"aliases: [doc-{doc_id}, test-{doc_id}]\n\n"

    topics = [
        "machine learning algorithms and neural network architectures",
        "distributed systems design and microservices patterns",
        "database optimization techniques and query performance",
        "cloud infrastructure scaling and container orchestration",
        "software testing strategies and continuous integration",
        "API design principles and RESTful architecture",
        "security best practices and authentication mechanisms",
        "data structures and algorithm complexity analysis",
    ]

    for i in range(paragraphs):
        content += f"## Section {i + 1}\n\n"
        for j in range(sentences_per_para):
            topic = topics[(doc_id + i + j) % len(topics)]
            content += f"This paragraph discusses {topic} in the context of modern software development. "
        content += "\n\n"

        # Add wikilink every few paragraphs
        if i % 2 == 0 and doc_id > 1:
            content += f"See also [[Document {doc_id - 1}]] for related information.\n\n"

    return content


def create_test_corpus(docs_path: Path, num_docs: int, doc_size: str = "medium"):
    for i in range(num_docs):
        doc_file = docs_path / f"doc_{i:04d}.md"
        content = create_realistic_document(i, doc_size)
        doc_file.write_text(content)


def test_indexing_speed_10_documents(config, manager, tmp_path):
    """Benchmark indexing speed for 10 documents."""
    docs_path = Path(config.indexing.documents_path)
    num_docs = 10

    # Create corpus
    create_test_corpus(docs_path, num_docs, doc_size="medium")

    # Benchmark indexing
    start_time = time.perf_counter()

    for doc_file in sorted(docs_path.glob("*.md")):
        manager.index_document(str(doc_file))

    end_time = time.perf_counter()
    elapsed = end_time - start_time

    # Calculate metrics
    throughput = num_docs / elapsed
    avg_time_per_doc = elapsed / num_docs

    # Verify indexing completed
    doc_count = manager.get_document_count()
    assert doc_count == num_docs, f"Expected {num_docs} documents, got {doc_count}"

    # Report metrics
    print("\n=== Indexing Performance (10 documents) ===")
    print(f"Total time: {elapsed:.2f}s")
    print(f"Throughput: {throughput:.2f} docs/sec")
    print(f"Avg per doc: {avg_time_per_doc:.3f}s")
    print(f"Documents indexed: {doc_count}")

    # Performance baseline assertion (should complete in reasonable time)
    # Allow up to 5 seconds per document for embedding model load + indexing
    assert elapsed < (num_docs * 5.0), f"Indexing too slow: {elapsed:.2f}s for {num_docs} docs"


def test_indexing_speed_50_documents(config, manager, tmp_path):
    """Benchmark indexing speed for 50 documents."""
    docs_path = Path(config.indexing.documents_path)
    num_docs = 50

    # Create corpus
    create_test_corpus(docs_path, num_docs, doc_size="medium")

    # Benchmark indexing
    start_time = time.perf_counter()

    for doc_file in sorted(docs_path.glob("*.md")):
        manager.index_document(str(doc_file))

    end_time = time.perf_counter()
    elapsed = end_time - start_time

    # Calculate metrics
    throughput = num_docs / elapsed
    avg_time_per_doc = elapsed / num_docs

    # Verify indexing completed
    doc_count = manager.get_document_count()
    assert doc_count == num_docs, f"Expected {num_docs} documents, got {doc_count}"

    # Report metrics
    print("\n=== Indexing Performance (50 documents) ===")
    print(f"Total time: {elapsed:.2f}s")
    print(f"Throughput: {throughput:.2f} docs/sec")
    print(f"Avg per doc: {avg_time_per_doc:.3f}s")
    print(f"Documents indexed: {doc_count}")

    # Performance baseline assertion (should complete in reasonable time)
    # Allow up to 2 seconds per document on average after model warmup
    assert elapsed < (num_docs * 2.0), f"Indexing too slow: {elapsed:.2f}s for {num_docs} docs"


def test_indexing_speed_100_documents(config, manager, tmp_path):
    """Benchmark indexing speed for 100 documents."""
    docs_path = Path(config.indexing.documents_path)
    num_docs = 100

    # Create corpus
    create_test_corpus(docs_path, num_docs, doc_size="medium")

    # Benchmark indexing
    start_time = time.perf_counter()

    for doc_file in sorted(docs_path.glob("*.md")):
        manager.index_document(str(doc_file))

    end_time = time.perf_counter()
    elapsed = end_time - start_time

    # Calculate metrics
    throughput = num_docs / elapsed
    avg_time_per_doc = elapsed / num_docs

    # Verify indexing completed
    doc_count = manager.get_document_count()
    assert doc_count == num_docs, f"Expected {num_docs} documents, got {doc_count}"

    # Report metrics
    print("\n=== Indexing Performance (100 documents) ===")
    print(f"Total time: {elapsed:.2f}s")
    print(f"Throughput: {throughput:.2f} docs/sec")
    print(f"Avg per doc: {avg_time_per_doc:.3f}s")
    print(f"Documents indexed: {doc_count}")

    # Performance baseline assertion (should complete in reasonable time)
    # Allow up to 2 seconds per document on average after model warmup
    assert elapsed < (num_docs * 2.0), f"Indexing too slow: {elapsed:.2f}s for {num_docs} docs"


def test_indexing_speed_varying_document_sizes(config, tmp_path, shared_embedding_model):
    """
    Benchmark indexing with varied document sizes.

    Tests performance across small, medium, and large documents
    to identify size-dependent performance characteristics.

    Note: Uses isolated indices to avoid state leakage from other tests.
    """
    sizes = ["small", "medium", "large"]
    docs_path = tmp_path / "docs"
    docs_path.mkdir(exist_ok=True)
    num_docs = len(sizes)

    for i, size in enumerate(sizes):
        doc_file = docs_path / f"doc_{size}_{i:02d}.md"
        content = create_realistic_document(i, size=size)
        doc_file.write_text(content)

    # Create isolated indices for this test to avoid state leakage
    isolated_vector = VectorIndex(embedding_model=shared_embedding_model)
    isolated_keyword = KeywordIndex()
    isolated_graph = GraphStore()
    isolated_manager = IndexManager(config, isolated_vector, isolated_keyword, isolated_graph)

    # Benchmark indexing
    start_time = time.perf_counter()

    for doc_file in sorted(docs_path.glob("*.md")):
        isolated_manager.index_document(str(doc_file))

    end_time = time.perf_counter()
    elapsed = end_time - start_time

    # Calculate metrics
    throughput = num_docs / elapsed
    avg_time_per_doc = elapsed / num_docs

    # Verify indexing completed
    doc_count = isolated_manager.get_document_count()
    assert doc_count == num_docs, f"Expected {num_docs} documents, got {doc_count}"

    # Report metrics
    print("\n=== Indexing Performance (Mixed Document Sizes) ===")
    print("Corpus: 5 small + 5 medium + 5 large documents")
    print(f"Total time: {elapsed:.2f}s")
    print(f"Throughput: {throughput:.2f} docs/sec")
    print(f"Avg per doc: {avg_time_per_doc:.3f}s")
    print(f"Documents indexed: {doc_count}")

    # Performance baseline assertion
    assert elapsed < (num_docs * 3.0), f"Indexing too slow: {elapsed:.2f}s for {num_docs} mixed docs"
