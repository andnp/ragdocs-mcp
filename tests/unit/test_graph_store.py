import pytest

from src.indices.graph import GraphStore


@pytest.fixture
def graph_store():
    return GraphStore()


def test_graph_store_add_node(graph_store):
    graph_store.add_node("doc1", {"title": "Document 1", "tags": ["test"]})

    # Verify node exists by checking it has no neighbors initially
    neighbors = graph_store.get_neighbors("doc1", depth=1)
    assert neighbors == []  # No edges yet, so no neighbors


def test_graph_store_add_edge(graph_store):
    graph_store.add_node("doc1", {})
    graph_store.add_node("doc2", {})

    graph_store.add_edge("doc1", "doc2", "link")

    # Verify edge exists by checking neighbors
    neighbors = graph_store.get_neighbors("doc1", depth=1)
    assert "doc2" in neighbors


def test_graph_store_add_edge_transclusion(graph_store):
    graph_store.add_node("doc1", {})
    graph_store.add_node("doc2", {})

    graph_store.add_edge("doc1", "doc2", "transclusion")

    # Verify edge exists by checking neighbors (edge type is internal detail)
    neighbors = graph_store.get_neighbors("doc1", depth=1)
    assert "doc2" in neighbors


def test_graph_store_remove_node(graph_store):
    graph_store.add_node("doc1", {})
    graph_store.add_node("doc2", {})
    graph_store.add_edge("doc1", "doc2", "link")

    # Verify edge exists before removal
    neighbors_before = graph_store.get_neighbors("doc1", depth=1)
    assert "doc2" in neighbors_before

    graph_store.remove_node("doc1")

    # Verify node removed: get_neighbors returns empty for nonexistent node
    neighbors_after = graph_store.get_neighbors("doc1", depth=1)
    assert neighbors_after == []


def test_graph_store_remove_nonexistent_node(graph_store):
    graph_store.remove_node("nonexistent")

    # Verify operation completed without error (no assertion needed)
    # Public API: get_neighbors returns empty list for nonexistent nodes
    neighbors = graph_store.get_neighbors("nonexistent", depth=1)
    assert neighbors == []


def test_graph_store_get_neighbors_depth_1(graph_store):
    graph_store.add_node("doc1", {})
    graph_store.add_node("doc2", {})
    graph_store.add_node("doc3", {})

    graph_store.add_edge("doc1", "doc2", "link")
    graph_store.add_edge("doc1", "doc3", "link")

    neighbors = graph_store.get_neighbors("doc1", depth=1)

    assert set(neighbors) == {"doc2", "doc3"}


def test_graph_store_get_neighbors_depth_2(graph_store):
    graph_store.add_node("doc1", {})
    graph_store.add_node("doc2", {})
    graph_store.add_node("doc3", {})
    graph_store.add_node("doc4", {})

    graph_store.add_edge("doc1", "doc2", "link")
    graph_store.add_edge("doc2", "doc3", "link")
    graph_store.add_edge("doc3", "doc4", "link")

    neighbors = graph_store.get_neighbors("doc1", depth=2)

    assert set(neighbors) == {"doc2", "doc3"}


def test_graph_store_get_neighbors_bidirectional(graph_store):
    graph_store.add_node("doc1", {})
    graph_store.add_node("doc2", {})
    graph_store.add_node("doc3", {})

    graph_store.add_edge("doc1", "doc2", "link")
    graph_store.add_edge("doc3", "doc1", "link")

    neighbors = graph_store.get_neighbors("doc1", depth=1)

    assert set(neighbors) == {"doc2", "doc3"}


def test_graph_store_get_neighbors_nonexistent_node(graph_store):
    neighbors = graph_store.get_neighbors("nonexistent", depth=1)

    assert neighbors == []


def test_graph_store_persist_and_load(tmp_path, graph_store):
    graph_store.add_node("doc1", {"title": "Document 1"})
    graph_store.add_node("doc2", {"title": "Document 2"})
    graph_store.add_edge("doc1", "doc2", "link")

    persist_path = tmp_path / "graph"
    graph_store.persist(persist_path)

    assert persist_path.exists()
    assert (persist_path / "graph.json").exists()

    new_store = GraphStore()
    new_store.load(persist_path)

    # Verify loaded data using public API
    neighbors_doc1 = new_store.get_neighbors("doc1", depth=1)
    assert "doc2" in neighbors_doc1

    # Verify both nodes exist
    neighbors_doc2 = new_store.get_neighbors("doc2", depth=1)
    assert "doc1" in neighbors_doc2  # Bidirectional check


def test_graph_store_load_nonexistent_path(tmp_path):
    store = GraphStore()
    nonexistent_path = tmp_path / "nonexistent"

    store.load(nonexistent_path)

    # Verify graph is empty by testing neighbor queries return empty
    store.add_node("test", {})
    neighbors = store.get_neighbors("test", depth=1)
    assert neighbors == []  # No edges in empty graph


def test_graph_store_complex_graph(graph_store):
    nodes = [f"doc{i}" for i in range(1, 6)]
    for node in nodes:
        graph_store.add_node(node, {"index": node})

    graph_store.add_edge("doc1", "doc2", "link")
    graph_store.add_edge("doc1", "doc3", "link")
    graph_store.add_edge("doc2", "doc4", "transclusion")
    graph_store.add_edge("doc3", "doc4", "link")
    graph_store.add_edge("doc4", "doc5", "link")

    neighbors_depth_1 = graph_store.get_neighbors("doc1", depth=1)
    assert set(neighbors_depth_1) == {"doc2", "doc3"}

    neighbors_depth_2 = graph_store.get_neighbors("doc1", depth=2)
    assert set(neighbors_depth_2) == {"doc2", "doc3", "doc4"}

    neighbors_depth_3 = graph_store.get_neighbors("doc1", depth=3)
    assert set(neighbors_depth_3) == {"doc2", "doc3", "doc4", "doc5"}


def test_graph_store_cyclic_graph(graph_store):
    graph_store.add_node("doc1", {})
    graph_store.add_node("doc2", {})
    graph_store.add_node("doc3", {})

    graph_store.add_edge("doc1", "doc2", "link")
    graph_store.add_edge("doc2", "doc3", "link")
    graph_store.add_edge("doc3", "doc1", "link")

    neighbors = graph_store.get_neighbors("doc1", depth=2)

    assert set(neighbors) == {"doc2", "doc3"}
