"""
Unit tests for GraphStore corruption handling.

Tests that malformed JSON files are handled gracefully with reinitialization.
"""

import json

import pytest

from src.indices.graph import GraphStore


@pytest.fixture
def graph_store():
    return GraphStore()


def test_graph_store_load_with_corrupted_graph_json(tmp_path, graph_store):
    """
    GraphStore gracefully handles corrupted graph.json file.

    When graph.json contains malformed JSON, the load() method should
    log a warning and reinitialize with an empty graph.
    """
    graph_store.add_node("doc1", {"title": "Test"})
    graph_store.add_edge("doc1", "doc2", "links_to", "context")

    persist_path = tmp_path / "graph_index"
    graph_store.persist(persist_path)

    graph_file = persist_path / "graph.json"
    graph_file.write_text("{ corrupted json")

    graph_store2 = GraphStore()
    graph_store2.load(persist_path)

    assert len(graph_store2) == 0
    assert graph_store2._communities == {}


def test_graph_store_load_with_corrupted_communities_json(tmp_path, graph_store):
    """
    GraphStore gracefully handles corrupted communities.json file.

    When communities.json is malformed, the graph should still load
    successfully with empty communities.
    """
    graph_store.add_node("doc1", {"title": "Test"})
    graph_store.add_node("doc2", {"title": "Test 2"})
    graph_store.add_edge("doc1", "doc2", "links_to")

    persist_path = tmp_path / "graph_index"
    graph_store.persist(persist_path)

    communities_file = persist_path / "communities.json"
    communities_file.write_text("not valid json at all")

    graph_store2 = GraphStore()
    graph_store2.load(persist_path)

    assert len(graph_store2) == 2
    assert graph_store2._communities == {}


def test_graph_store_load_with_missing_graph_file(tmp_path):
    """
    GraphStore handles missing graph.json file.

    When graph.json doesn't exist, load should initialize empty graph.
    """
    persist_path = tmp_path / "empty_graph_index"
    persist_path.mkdir()

    graph_store = GraphStore()
    graph_store.load(persist_path)

    assert len(graph_store) == 0
    assert graph_store._communities == {}


def test_graph_store_load_with_empty_json_files(tmp_path):
    """
    GraphStore handles empty but valid JSON files.

    Tests loading from directory with minimal valid JSON.
    """
    persist_path = tmp_path / "empty_index"
    persist_path.mkdir()

    graph_data = {"directed": True, "multigraph": False, "graph": {}, "nodes": [], "edges": []}
    (persist_path / "graph.json").write_text(json.dumps(graph_data))
    (persist_path / "communities.json").write_text("{}")

    graph_store = GraphStore()
    graph_store.load(persist_path)

    assert len(graph_store) == 0
    assert graph_store._communities == {}


def test_graph_store_recovery_allows_new_nodes(tmp_path):
    """
    After corruption recovery, new nodes can be added successfully.

    Tests the full cycle: create graph, persist, corrupt, detect
    corruption during load, reinitialize, then add new nodes.
    """
    graph_store = GraphStore()
    graph_store.add_node("original", {"title": "Original"})

    persist_path = tmp_path / "recovery_test"
    graph_store.persist(persist_path)

    graph_file = persist_path / "graph.json"
    graph_file.write_text("{ incomplete")

    graph_store2 = GraphStore()
    graph_store2.load(persist_path)

    assert len(graph_store2) == 0

    graph_store2.add_node("new_doc", {"title": "New After Recovery"})
    graph_store2.add_edge("new_doc", "other_doc", "links_to")

    assert len(graph_store2) == 2
    assert graph_store2.has_node("new_doc")


def test_graph_store_corrupted_graph_returns_empty_search(tmp_path):
    """
    Search on corrupted graph returns empty results.

    Tests that search works on reinitialized empty graph.
    """
    graph_store = GraphStore()
    graph_store.add_node("doc1", {"content": "searchable content"})

    persist_path = tmp_path / "search_test"
    graph_store.persist(persist_path)

    graph_file = persist_path / "graph.json"
    graph_file.write_text("corrupted")

    graph_store2 = GraphStore()
    graph_store2.load(persist_path)

    results = graph_store2.search("searchable", limit=5)
    assert results == []


def test_graph_store_persist_after_corruption_recovery(tmp_path):
    """
    Persist works correctly after corruption recovery.

    Tests that graph can be persisted again after recovering from corruption.
    """
    graph_store = GraphStore()
    graph_store.add_node("original", {"title": "Original"})

    persist_path = tmp_path / "persist_test"
    graph_store.persist(persist_path)

    graph_file = persist_path / "graph.json"
    graph_file.write_text("{ bad json")

    graph_store2 = GraphStore()
    graph_store2.load(persist_path)

    graph_store2.add_node("recovered", {"title": "After Recovery"})

    new_persist_path = tmp_path / "new_persist"
    graph_store2.persist(new_persist_path)

    graph_store3 = GraphStore()
    graph_store3.load(new_persist_path)

    assert graph_store3.has_node("recovered")
    assert len(graph_store3) == 1


def test_graph_store_truncated_json(tmp_path):
    """
    GraphStore handles truncated JSON (partial write scenario).

    Simulates a crash during write that leaves partial JSON.
    """
    persist_path = tmp_path / "truncated_test"
    persist_path.mkdir()

    truncated_json = '{"directed": true, "nodes": [{"id": "doc1"'
    (persist_path / "graph.json").write_text(truncated_json)

    graph_store = GraphStore()
    graph_store.load(persist_path)

    assert len(graph_store) == 0


def test_graph_store_invalid_json_types(tmp_path):
    """
    GraphStore handles JSON with unexpected types.

    Tests that valid JSON with wrong structure (string instead of dict)
    triggers corruption recovery and allows subsequent operations.
    """
    persist_path = tmp_path / "wrong_types"
    persist_path.mkdir()

    (persist_path / "graph.json").write_text('"just a string"')

    graph_store = GraphStore()
    graph_store.load(persist_path)

    assert len(graph_store) == 0

    graph_store.add_node("test", {})
    assert graph_store.has_node("test")
    assert len(graph_store) == 1
