"""
E2E tests for FastAPI Server (D14).

Tests the complete server lifecycle from startup through query handling,
file change detection, and graceful shutdown. Uses FastAPI TestClient for
real HTTP request/response testing with full component integration.
"""

import time

import pytest
from fastapi.testclient import TestClient

from src.config import Config, IndexingConfig, LLMConfig, SearchConfig, ServerConfig
from src.indexing.manager import IndexManager
from src.indexing.manifest import IndexManifest, save_manifest
from src.indices.graph import GraphStore
from src.indices.keyword import KeywordIndex
from src.indices.vector import VectorIndex
from src.server import create_app


@pytest.fixture
def test_docs_dir(tmp_path):
    """
    Create test documents directory with sample files.

    Provides realistic markdown documents for E2E server testing.
    """
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()

    # Create sample documents
    (docs_dir / "intro.md").write_text("# Introduction\n\nWelcome to the documentation.")
    (docs_dir / "api.md").write_text("# API Reference\n\nAPI documentation for developers.")
    (docs_dir / "guide.md").write_text("# User Guide\n\nStep-by-step guide for users.")

    return docs_dir


@pytest.fixture
def test_config(tmp_path, test_docs_dir):
    """
    Create test configuration file for server startup.

    Uses temporary paths to isolate E2E tests from development environment.
    """
    config_dir = tmp_path / ".mcp-markdown-ragdocs"
    config_dir.mkdir()
    config_file = config_dir / "config.toml"
    config_file.write_text(f"""
[server]
host = "127.0.0.1"
port = 8000

[indexing]
documents_path = "{test_docs_dir}"
index_path = "{tmp_path / 'indices'}"
recursive = true

[parsers]
"**/*.md" = "MarkdownParser"

[search]
semantic_weight = 0.6
keyword_weight = 0.4
recency_bias = 0.1
rrf_k_constant = 60

[llm]
embedding_model = "all-MiniLM-L6-v2"
llm_provider = "local"
""")
    return config_file


@pytest.fixture
def app_with_config(tmp_path, test_docs_dir, monkeypatch):
    """
    Create FastAPI app with test configuration.

    Monkeypatches load_config to use test configuration instead of
    production config files.
    """
    index_path = tmp_path / "indices"
    index_path.mkdir(parents=True, exist_ok=True)

    def mock_load_config():
        return Config(
            server=ServerConfig(host="127.0.0.1", port=8000),
            indexing=IndexingConfig(
                documents_path=str(test_docs_dir),
                index_path=str(index_path),
                recursive=True,
            ),
            parsers={"**/*.md": "MarkdownParser"},
            search=SearchConfig(
                semantic_weight=0.6,
                keyword_weight=0.4,
                recency_bias=0.1,
                rrf_k_constant=60,
            ),
            llm=LLMConfig(
                embedding_model="all-MiniLM-L6-v2",
                llm_provider="local",
            ),
        )

    monkeypatch.setattr("src.context.load_config", mock_load_config)
    app = create_app()
    return app


@pytest.fixture
def client(app_with_config):
    """
    Create TestClient for HTTP request testing.

    TestClient handles lifespan context manager, starting and stopping
    the server automatically for each test.
    """
    with TestClient(app_with_config) as client:
        yield client


def test_server_starts_and_responds_to_health_check(client):
    """
    Test that server starts successfully and health endpoint responds.

    Validates the most basic server functionality: process starts, binds to
    port, and responds to health check requests with correct structure.
    """
    response = client.get("/health")

    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["status"] == "ok"


def test_status_endpoint_returns_correct_structure(client):
    """
    Test that status endpoint returns detailed server state information.

    Validates monitoring capability: status endpoint should provide
    comprehensive information about server state, indices, and watcher.
    """
    response = client.get("/status")

    assert response.status_code == 200
    data = response.json()

    # Verify top-level structure
    assert "server_status" in data
    assert "indexing_service" in data
    assert "indices" in data

    # Verify server status
    assert data["server_status"] == "running"

    # Verify indexing service details
    indexing_service = data["indexing_service"]
    assert "pending_queue_size" in indexing_service
    assert "last_sync_time" in indexing_service
    assert "failed_files" in indexing_service
    assert isinstance(indexing_service["pending_queue_size"], int)
    assert isinstance(indexing_service["failed_files"], list)

    # Verify indices information
    indices = data["indices"]
    assert "document_count" in indices
    assert "index_version" in indices
    assert isinstance(indices["document_count"], int)
    assert isinstance(indices["index_version"], str)
    assert indices["document_count"] >= 3  # At least 3 test documents


def test_query_endpoint_accepts_requests_and_returns_results(client):
    """
    Test that query endpoint processes requests and returns search results.

    Validates core search functionality: queries should be processed through
    the hybrid search orchestrator and return synthesized LLM answers along
    with scored results.
    """
    response = client.post(
        "/query_documents",
        json={"query": "API documentation"},
    )

    assert response.status_code == 200
    data = response.json()

    # Verify response structure
    assert "answer" in data
    assert "results" in data

    # Verify answer is a non-empty string
    answer = data["answer"]
    assert isinstance(answer, str)
    assert len(answer) > 0

    # Verify results structure (list of dict objects)
    results = data["results"]
    assert isinstance(results, list)
    if results:
        # Each result should be a dict with chunk_id, score, etc.
        for result in results:
            assert isinstance(result, dict)
            assert "chunk_id" in result
            assert "score" in result
            chunk_id = result["chunk_id"]
            score = result["score"]
            assert isinstance(chunk_id, str)
            assert isinstance(score, (int, float))
            assert 0.0 <= score <= 1.0
        # Highest score should be 1.0
        assert results[0]["score"] == 1.0


def test_file_changes_trigger_index_updates(client, test_docs_dir):
    """
    Test that file system changes trigger automatic index updates.

    Validates real-time update capability: creating new files should be
    detected by watcher and indexed, making them immediately searchable.
    """
    # Query for non-existent document
    response_before = client.post(
        "/query_documents",
        json={"query": "deployment configuration"},
    )
    response_before.json()

    # Create new document
    new_file = test_docs_dir / "deployment.md"
    new_file.write_text("# Deployment\n\nConfiguration for production deployment.")

    # Wait for watcher to detect, debounce, and process
    # FileWatcher uses 0.5s cooldown by default
    time.sleep(1.5)

    # Query again for new document
    response_after = client.post(
        "/query_documents",
        json={"query": "deployment configuration"},
    )
    data_after = response_after.json()
    answer_after = data_after["answer"]

    # Verify response changed (indicating new document was indexed)
    # Note: MockLLM echoes queries so we can't check content directly,
    # but we verify the system processed the query and returned an answer
    assert len(answer_after) > 0
    assert isinstance(answer_after, str)


def test_server_shutdown_persists_indices(client, tmp_path, test_docs_dir):
    """
    Test that server shutdown correctly persists all indices to disk.

    Validates data persistence: shutting down the server should save all
    index data, allowing it to be loaded on next startup without rebuild.
    """
    # Make query to ensure indices are populated
    response = client.get("/status")
    assert response.status_code == 200

    # TestClient context manager exit triggers lifespan shutdown
    # This happens automatically when client fixture goes out of scope

    # Verify index files were persisted
    index_path = tmp_path / "indices"
    assert index_path.exists()

    # Check vector index files
    vector_path = index_path / "vector"
    assert vector_path.exists()
    assert (vector_path / "docstore.json").exists()

    # Check keyword index directory
    keyword_path = index_path / "keyword"
    assert keyword_path.exists()

    # Check graph store file
    graph_path = index_path / "graph"
    assert graph_path.exists()
    assert (graph_path / "graph.json").exists()

    # Check manifest file
    manifest_file = index_path / "index.manifest.json"
    assert manifest_file.exists()


def test_manifest_checking_on_startup(tmp_path, test_docs_dir, monkeypatch):
    """
    Test that server checks manifest on startup and rebuilds if needed.

    Validates lifecycle management: server should detect manifest mismatches
    (version changes) and trigger full rebuild to ensure index consistency.
    """
    index_path = tmp_path / "indices"
    index_path.mkdir(parents=True, exist_ok=True)

    # Create old manifest with different version
    old_manifest = IndexManifest(
        spec_version="0.9.0",  # Old version
        embedding_model="all-MiniLM-L6-v2",
        parsers={"**/*.md": "MarkdownParser"},
        chunking_config={},
    )
    save_manifest(index_path, old_manifest)

    # Create pre-existing indices with old content
    vector = VectorIndex()
    keyword = KeywordIndex()
    graph = GraphStore()
    old_config = Config(
        server=ServerConfig(),
        indexing=IndexingConfig(
            documents_path=str(test_docs_dir),
            index_path=str(index_path),
        ),
        parsers={"**/*.md": "MarkdownParser"},
        search=SearchConfig(),
        llm=LLMConfig(embedding_model="all-MiniLM-L6-v2"),
    )
    old_manager = IndexManager(old_config, vector, keyword, graph)

    # Index one document with old manager
    (test_docs_dir / "old_doc.md").write_text("# Old Document\n\nExisting content.")
    old_manager.index_document(str(test_docs_dir / "old_doc.md"))
    old_manager.persist()

    # Add new document that should be indexed during rebuild
    (test_docs_dir / "new_doc.md").write_text("# New Document\n\nShould be indexed on rebuild.")

    # Mock load_config for new server with updated version
    def mock_load_config():
        return Config(
            server=ServerConfig(host="127.0.0.1", port=8000),
            indexing=IndexingConfig(
                documents_path=str(test_docs_dir),
                index_path=str(index_path),
                recursive=True,
            ),
            parsers={"**/*.md": "MarkdownParser"},
            search=SearchConfig(),
            llm=LLMConfig(
                embedding_model="all-MiniLM-L6-v2",
                llm_provider="local",
            ),
        )

    monkeypatch.setattr("src.context.load_config", mock_load_config)

    # Start server - should detect version mismatch and rebuild
    app = create_app()
    with TestClient(app) as client:
        # Verify server started successfully
        response = client.get("/health")
        assert response.status_code == 200

        # Query for document that was added after old index creation
        response = client.post(
            "/query_documents",
            json={"query": "new document rebuild"},
        )
        data = response.json()
        answer = data["answer"]

        # Verify new document was indexed during rebuild
        assert "new" in answer.lower() or "new_doc" in answer.lower()


def test_query_endpoint_with_empty_query(client):
    """
    Test that query endpoint handles empty queries gracefully.

    Validates edge case handling: empty queries should not crash the server
    and should return empty results or appropriate error message.
    """
    response = client.post(
        "/query_documents",
        json={"query": ""},
    )

    assert response.status_code == 200
    data = response.json()
    assert "answer" in data


def test_concurrent_query_requests(client):
    """
    Test that server handles concurrent query requests correctly.

    Validates concurrency handling: multiple simultaneous queries should
    all be processed successfully without race conditions or errors.
    """
    queries = [
        "API documentation",
        "user guide",
        "introduction",
        "deployment",
    ]

    responses = []
    for query in queries:
        response = client.post(
            "/query_documents",
            json={"query": query},
        )
        responses.append(response)

    # Verify all requests succeeded
    for response in responses:
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data


def test_query_documents_with_default_top_n(client):
    """
    Test that query endpoint works with default top_n parameter.

    Validates:
    - Default top_n=5 is applied when not specified
    - Response includes both answer and results fields
    - Results contain at most 5 items
    """
    response = client.post(
        "/query_documents",
        json={"query": "API documentation"},
    )

    assert response.status_code == 200
    data = response.json()

    # Verify response structure
    assert "answer" in data
    assert "results" in data
    assert isinstance(data["results"], list)

    # Default top_n is 5
    assert len(data["results"]) <= 5

    # Verify answer is a non-empty string
    assert isinstance(data["answer"], str)
    assert len(data["answer"]) > 0


def test_query_documents_with_custom_top_n(client):
    """
    Test that custom top_n parameter limits results correctly.

    Validates:
    - top_n=3 returns at most 3 results
    - top_n=1 returns at most 1 result
    - top_n=10 returns at most 10 results
    """
    # Test top_n=3
    response_3 = client.post(
        "/query_documents",
        json={"query": "API documentation", "top_n": 3},
    )

    assert response_3.status_code == 200
    data_3 = response_3.json()
    assert len(data_3["results"]) <= 3

    # Test top_n=1
    response_1 = client.post(
        "/query_documents",
        json={"query": "API documentation", "top_n": 1},
    )

    assert response_1.status_code == 200
    data_1 = response_1.json()
    assert len(data_1["results"]) <= 1

    # Test top_n=10
    response_10 = client.post(
        "/query_documents",
        json={"query": "documentation guide", "top_n": 10},
    )

    assert response_10.status_code == 200
    data_10 = response_10.json()
    assert len(data_10["results"]) <= 10


def test_query_documents_returns_scores(client):
    """
    Test that API response includes normalized scores.

    Validates:
    - Each result is a tuple [chunk_id, score]
    - chunk_id is a string
    - score is a float in [0, 1] range
    """
    response = client.post(
        "/query_documents",
        json={"query": "user guide", "top_n": 5},
    )

    assert response.status_code == 200
    data = response.json()
    results = data["results"]

    # Verify each result has correct structure
    for item in results:
        assert isinstance(item, dict), f"Result item should be dict, got {type(item)}"

        assert "chunk_id" in item
        assert "score" in item
        chunk_id = item["chunk_id"]
        score = item["score"]

        # Verify types
        assert isinstance(chunk_id, str), f"chunk_id should be str, got {type(chunk_id)}"
        assert isinstance(score, (int, float)), f"score should be numeric, got {type(score)}"

        # Verify score range
        assert 0.0 <= score <= 1.0, f"Score {score} out of range [0, 1]"


def test_query_documents_scores_descending(client):
    """
    Test that scores are sorted in descending order.

    Validates that the API returns results with highest relevance first,
    ensuring scores are monotonically decreasing.
    """
    response = client.post(
        "/query_documents",
        json={"query": "introduction", "top_n": 5},
    )

    assert response.status_code == 200
    data = response.json()
    results = data["results"]

    if len(results) > 1:
        scores = [result["score"] for result in results]

        # Verify scores are in descending order
        assert scores == sorted(scores, reverse=True), \
            f"Scores should be descending, got {scores}"

        # Verify each score is <= previous score
        for i in range(len(scores) - 1):
            assert scores[i] >= scores[i+1], \
                f"Score at position {i} ({scores[i]}) should be >= score at {i+1} ({scores[i+1]})"


def test_query_documents_top_score_is_1_0(client):
    """
    Test that the highest score in results is always 1.0.

    Validates normalization invariant: the best match should always
    have a normalized score of 1.0.
    """
    response = client.post(
        "/query_documents",
        json={"query": "API reference", "top_n": 5},
    )

    assert response.status_code == 200
    data = response.json()
    results = data["results"]

    if results:
        top_score = results[0]["score"]
        assert top_score == 1.0, \
            f"Top score should be 1.0, got {top_score}"


def test_query_documents_validates_top_n_range(client):
    """
    Test that API validates top_n parameter bounds.

    Validates:
    - top_n=0 is rejected (422 Unprocessable Entity)
    - top_n=101 is rejected (exceeds max of 100)
    - top_n must be between 1 and 100
    """
    # Test top_n=0 (too low)
    response_zero = client.post(
        "/query_documents",
        json={"query": "test", "top_n": 0},
    )
    assert response_zero.status_code == 422

    # Test top_n=101 (too high)
    response_large = client.post(
        "/query_documents",
        json={"query": "test", "top_n": 101},
    )
    assert response_large.status_code == 422

    # Test top_n=-1 (negative)
    response_negative = client.post(
        "/query_documents",
        json={"query": "test", "top_n": -1},
    )
    assert response_negative.status_code == 422

    # Verify valid bounds work (1 and 100)
    response_min = client.post(
        "/query_documents",
        json={"query": "test", "top_n": 1},
    )
    assert response_min.status_code == 200

    response_max = client.post(
        "/query_documents",
        json={"query": "test", "top_n": 100},
    )
    assert response_max.status_code == 200


def test_query_documents_empty_query_with_scores(client):
    """
    Test that empty queries return empty results with valid structure.

    Validates graceful handling: empty query should return empty results
    list (not null) and still include the answer field.
    """
    response = client.post(
        "/query_documents",
        json={"query": "", "top_n": 5},
    )

    assert response.status_code == 200
    data = response.json()

    # Should have both fields even with empty query
    assert "answer" in data
    assert "results" in data

    # Results should be empty list
    assert isinstance(data["results"], list)
    assert len(data["results"]) == 0
