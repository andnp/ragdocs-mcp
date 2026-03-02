"""
Unit tests for GraphStore corruption handling.

Tests that malformed legacy JSON files are handled gracefully during migration,
and that SQLite-backed GraphStore self-heals after corruption.
"""

import json

import pytest

from src.indices.graph import GraphStore
from src.storage.db import DatabaseManager


@pytest.fixture
def graph_store(tmp_path):
    db = DatabaseManager(tmp_path / "test.db")
    return GraphStore(db)


def test_graph_store_load_with_corrupted_graph_json(tmp_path):
    """
    GraphStore gracefully handles corrupted graph.json during migration.

    When graph.json contains malformed JSON, the load() method should
    log a warning and skip migration (leaving the graph empty).
    """
    persist_path = tmp_path / "graph_index"
    persist_path.mkdir()

    graph_file = persist_path / "graph.json"
    graph_file.write_text("{ corrupted json")

    db = DatabaseManager(tmp_path / "test.db")
    graph_store = GraphStore(db)
    graph_store.load(persist_path)

    assert len(graph_store) == 0
    assert graph_store._communities == {}


def test_graph_store_load_with_corrupted_communities_json(tmp_path):
    """
    GraphStore gracefully handles corrupted communities.json during migration.

    communities.json is no longer used by the SQLite-backed GraphStore,
    so corrupted files are simply ignored during migration.
    """
    persist_path = tmp_path / "graph_index"
    persist_path.mkdir()

    # Create a valid graph.json with data
    graph_data = {
        "directed": True,
        "multigraph": False,
        "graph": {},
        "nodes": [{"id": "doc1"}, {"id": "doc2"}],
        "links": [{"source": "doc1", "target": "doc2", "edge_type": "links_to"}],
    }
    (persist_path / "graph.json").write_text(json.dumps(graph_data))
    (persist_path / "communities.json").write_text("not valid json at all")

    db = DatabaseManager(tmp_path / "test.db")
    graph_store = GraphStore(db)
    graph_store.load(persist_path)

    assert len(graph_store) == 2
    assert graph_store._communities == {}


def test_graph_store_load_with_missing_graph_file(tmp_path):
    """
    GraphStore handles missing graph.json file.

    When graph.json doesn't exist, load should leave graph empty.
    """
    persist_path = tmp_path / "empty_graph_index"
    persist_path.mkdir()

    db = DatabaseManager(tmp_path / "test.db")
    graph_store = GraphStore(db)
    graph_store.load(persist_path)

    assert len(graph_store) == 0
    assert graph_store._communities == {}


def test_graph_store_load_with_empty_json_files(tmp_path):
    """
    GraphStore handles empty but valid JSON files during migration.
    """
    persist_path = tmp_path / "empty_index"
    persist_path.mkdir()

    graph_data = {
        "directed": True,
        "multigraph": False,
        "graph": {},
        "nodes": [],
        "edges": [],
    }
    (persist_path / "graph.json").write_text(json.dumps(graph_data))
    (persist_path / "communities.json").write_text("{}")

    db = DatabaseManager(tmp_path / "test.db")
    graph_store = GraphStore(db)
    graph_store.load(persist_path)

    assert len(graph_store) == 0
    assert graph_store._communities == {}


def test_graph_store_recovery_allows_new_nodes(tmp_path):
    """
    After migration from corrupted JSON, new nodes can be added.
    """
    persist_path = tmp_path / "recovery_test"
    persist_path.mkdir()
    (persist_path / "graph.json").write_text("{ incomplete")

    db = DatabaseManager(tmp_path / "test.db")
    graph_store = GraphStore(db)
    graph_store.load(persist_path)

    assert len(graph_store) == 0

    graph_store.add_node("new_doc", {"title": "New After Recovery"})
    graph_store.add_edge("new_doc", "other_doc", "links_to")

    assert len(graph_store) == 2
    assert graph_store.has_node("new_doc")


def test_graph_store_corrupted_graph_returns_empty_search(tmp_path):
    """
    Search on graph loaded from corrupted JSON returns empty results.
    """
    persist_path = tmp_path / "search_test"
    persist_path.mkdir()
    (persist_path / "graph.json").write_text("corrupted")

    db = DatabaseManager(tmp_path / "test.db")
    graph_store = GraphStore(db)
    graph_store.load(persist_path)

    results = graph_store.search("searchable", limit=5)
    assert results == []


def test_graph_store_persist_after_corruption_recovery(tmp_path):
    """
    Data persists in SQLite across GraphStore instances sharing the same DB.
    """
    db = DatabaseManager(tmp_path / "shared.db")
    graph_store = GraphStore(db)
    graph_store.add_node("recovered", {"title": "After Recovery"})

    graph_store2 = GraphStore(db)
    assert graph_store2.has_node("recovered")
    assert len(graph_store2) == 1


def test_graph_store_truncated_json(tmp_path):
    """
    GraphStore handles truncated JSON (partial write scenario).
    """
    persist_path = tmp_path / "truncated_test"
    persist_path.mkdir()

    truncated_json = '{"directed": true, "nodes": [{"id": "doc1"'
    (persist_path / "graph.json").write_text(truncated_json)

    db = DatabaseManager(tmp_path / "test.db")
    graph_store = GraphStore(db)
    graph_store.load(persist_path)

    assert len(graph_store) == 0


def test_graph_store_invalid_json_types(tmp_path):
    """
    GraphStore handles JSON with unexpected types.

    Tests that valid JSON with wrong structure (string instead of dict)
    is skipped during migration and allows subsequent operations.
    """
    persist_path = tmp_path / "wrong_types"
    persist_path.mkdir()

    (persist_path / "graph.json").write_text('"just a string"')

    db = DatabaseManager(tmp_path / "test.db")
    graph_store = GraphStore(db)
    graph_store.load(persist_path)

    assert len(graph_store) == 0

    graph_store.add_node("test", {})
    assert graph_store.has_node("test")
    assert len(graph_store) == 1
