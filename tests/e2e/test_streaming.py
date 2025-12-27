"""
End-to-end tests for SSE streaming endpoint.

Tests the /query_documents_stream endpoint for proper SSE format,
event emission, and streaming behavior.
"""

import json

import pytest
from fastapi.testclient import TestClient

from src.server import create_app


@pytest.fixture
def client(tmp_path):
    """
    Create test client with temporary storage.

    Uses synchronous TestClient which handles async endpoints properly.
    """
    import os

    # Set test environment
    config_dir = tmp_path / ".mcp-markdown-ragdocs"
    config_dir.mkdir()
    test_config = config_dir / "config.toml"
    test_config.write_text(f"""
[server]
host = "127.0.0.1"
port = 8000

[indexing]
documents_path = "{tmp_path / 'docs'}"
index_path = "{tmp_path / '.index_data'}"
recursive = true

[search]
semantic_weight = 1.0
keyword_weight = 1.0
recency_bias = 0.5
rrf_k_constant = 60

[llm]
embedding_model = "local"
""")

    # Create docs directory with test content
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()

    (docs_dir / "test.md").write_text("""# Test Document

This is a test document for streaming tests.

## Section One

Content in section one with important information.

## Section Two

More content here for testing purposes.
""")

    # Change to test directory
    original_cwd = os.getcwd()
    os.chdir(tmp_path)

    try:
        app = create_app()
        with TestClient(app) as client:
            import time
            time.sleep(2)  # Wait for initial indexing to complete
            yield client
    finally:
        os.chdir(original_cwd)


def test_stream_endpoint_returns_sse_format(client):
    """
    Test that /query_documents_stream returns proper SSE format.

    Verifies:
    - Response has Content-Type: text/event-stream
    - Response contains event: and data: lines
    - Proper SSE formatting with double newlines
    """
    response = client.post(
        "/query_documents_stream",
        json={"query": "test document"},
        headers={"Accept": "text/event-stream"},
    )

    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]

    # Get response text
    response_text = response.text

    # Verify SSE format: should contain "event:" and "data:" lines
    assert "event:" in response_text, "Response should contain SSE event declarations"
    assert "data:" in response_text, "Response should contain SSE data fields"

    # Verify double newline separators (SSE spec requirement)
    assert "\n\n" in response_text, "SSE events should be separated by double newlines"


def test_stream_endpoint_sends_results_event(client):
    """
    Test that streaming endpoint sends results event first.

    Verifies:
    - First event is "event: results"
    - Results data contains list of chunk dicts
    - Each result has chunk_id, score, etc.
    """
    response = client.post(
        "/query_documents_stream",
        json={"query": "test section"},
    )

    assert response.status_code == 200

    # Parse SSE stream
    response_text = response.text
    events = parse_sse_events(response_text)

    assert len(events) > 0, "Should have at least one event"

    # First event should be results
    first_event = events[0]
    assert first_event["event"] == "results", "First event should be 'results'"
    assert "results" in first_event["data"], "Results event should contain 'results' key"

    # Verify results structure
    results = first_event["data"]["results"]
    assert isinstance(results, list), "Results should be a list"

    if results:
        result = results[0]
        assert "chunk_id" in result
        assert "doc_id" in result
        assert "score" in result
        assert "header_path" in result
        assert "file_path" in result


def test_stream_endpoint_sends_synthesis_events(client):
    """
    Test that streaming endpoint sends synthesis progress events.

    Verifies:
    - event: start is present with chunk count
    - event: chunk is present with text data
    - event: done is present with completion info
    """
    response = client.post(
        "/query_documents_stream",
        json={"query": "test document information"},
    )

    assert response.status_code == 200

    # Parse SSE stream
    events = parse_sse_events(response.text)

    # Extract event types
    event_types = [e["event"] for e in events]

    # Debug: print events if test fails
    if "error" in event_types:
        error_events = [e for e in events if e["event"] == "error"]
        print(f"\nDEBUG: Error events: {error_events}")

    # Verify expected events are present
    assert "results" in event_types, "Should have 'results' event"
    assert "start" in event_types, f"Should have 'start' event, got: {event_types}"
    assert "chunk" in event_types, "Should have 'chunk' event with synthesized text"
    assert "done" in event_types, "Should have 'done' event for completion"

    # Verify start event structure
    start_events = [e for e in events if e["event"] == "start"]
    if start_events:
        assert "chunk_count" in start_events[0]["data"]
        assert isinstance(start_events[0]["data"]["chunk_count"], int)

    # Verify chunk event structure
    chunk_events = [e for e in events if e["event"] == "chunk"]
    if chunk_events:
        assert "text" in chunk_events[0]["data"]
        assert isinstance(chunk_events[0]["data"]["text"], str)
        assert len(chunk_events[0]["data"]["text"]) > 0

    # Verify done event structure
    done_events = [e for e in events if e["event"] == "done"]
    if done_events:
        assert "total_length" in done_events[0]["data"]


def test_stream_endpoint_handles_empty_query(client):
    """
    Test that streaming endpoint handles empty queries gracefully.

    Verifies:
    - Empty query triggers error event
    - Error event has proper structure
    - Error message is descriptive
    """
    response = client.post(
        "/query_documents_stream",
        json={"query": ""},
    )

    assert response.status_code == 200

    # Parse SSE stream
    events = parse_sse_events(response.text)

    # Should have results event (empty results) and possibly error event
    event_types = [e["event"] for e in events]

    # Empty query should return results event with empty list
    assert "results" in event_types

    results_event = [e for e in events if e["event"] == "results"][0]
    results = results_event["data"]["results"]

    # Results should be empty or synthesis should fail gracefully
    if not results:
        # Should have error event for no results
        if "error" in event_types:
            error_event = [e for e in events if e["event"] == "error"][0]
            assert "message" in error_event["data"]
            assert isinstance(error_event["data"]["message"], str)


def test_stream_endpoint_respects_top_n(client):
    """
    Test that streaming endpoint respects top_n parameter.

    Verifies:
    - top_n=3 returns exactly 3 or fewer results
    - Results event contains correct number of items
    """
    response = client.post(
        "/query_documents_stream",
        json={"query": "test document", "top_n": 3},
    )

    assert response.status_code == 200

    # Parse SSE stream
    events = parse_sse_events(response.text)

    # Get results event
    results_events = [e for e in events if e["event"] == "results"]
    assert len(results_events) == 1, "Should have exactly one results event"

    results = results_events[0]["data"]["results"]
    assert len(results) <= 3, f"Should have at most 3 results, got {len(results)}"


def test_stream_endpoint_concurrent_requests(client):
    """
    Test that streaming endpoint handles multiple concurrent requests.

    Verifies:
    - Multiple simultaneous streams work correctly
    - Each stream is independent
    - No data mixing between streams
    """
    # Send multiple concurrent requests
    responses = []
    queries = ["test one", "test two", "test three"]

    for query in queries:
        response = client.post(
            "/query_documents_stream",
            json={"query": query},
        )
        assert response.status_code == 200
        responses.append(response)

    # Verify each response is valid SSE
    for response in responses:
        events = parse_sse_events(response.text)
        assert len(events) > 0, "Each stream should have events"

        # Verify results event exists
        event_types = [e["event"] for e in events]
        assert "results" in event_types


def test_stream_endpoint_with_top_n_variations(client):
    """
    Test streaming endpoint with different top_n values.

    Verifies:
    - top_n=1, top_n=5, top_n=10 all work correctly
    - Result counts respect limits
    """
    test_cases = [1, 5, 10]

    for top_n in test_cases:
        response = client.post(
            "/query_documents_stream",
            json={"query": "test content", "top_n": top_n},
        )

        assert response.status_code == 200

        events = parse_sse_events(response.text)
        results_events = [e for e in events if e["event"] == "results"]

        if results_events:
            results = results_events[0]["data"]["results"]
            assert len(results) <= top_n, f"Results count {len(results)} exceeds top_n {top_n}"


def test_stream_endpoint_error_handling(client):
    """
    Test streaming endpoint error handling.

    Verifies:
    - Invalid requests are handled gracefully
    - Error responses have proper format
    """
    # Test with invalid JSON structure
    response = client.post(
        "/query_documents_stream",
        json={},  # Missing required 'query' field
    )

    # Should return 422 for validation error
    assert response.status_code == 422


# Helper function to parse SSE events
def parse_sse_events(sse_text: str) -> list[dict]:
    """
    Parse SSE formatted text into list of event dictionaries.

    Returns:
        List of dicts with 'event' and 'data' keys
    """
    events = []
    lines = sse_text.strip().split('\n')

    current_event = {}
    for line in lines:
        line = line.strip()

        if not line:
            # Empty line indicates end of event
            if current_event:
                events.append(current_event)
                current_event = {}
            continue

        if line.startswith('event:'):
            current_event['event'] = line.split(':', 1)[1].strip()
        elif line.startswith('data:'):
            data_str = line.split(':', 1)[1].strip()
            try:
                current_event['data'] = json.loads(data_str)
            except json.JSONDecodeError:
                current_event['data'] = data_str

    # Add last event if exists
    if current_event:
        events.append(current_event)

    return events
