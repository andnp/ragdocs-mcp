import json
from pathlib import Path

import networkx as nx


class GraphStore:
    def __init__(self):
        self._graph: nx.DiGraph = nx.DiGraph()

    def add_node(self, doc_id: str, metadata: dict) -> None:
        self._graph.add_node(doc_id, **metadata)

    def add_edge(self, source: str, target: str, edge_type: str) -> None:
        self._graph.add_edge(source, target, edge_type=edge_type)

    def remove_node(self, doc_id: str) -> None:
        if doc_id in self._graph:
            self._graph.remove_node(doc_id)

    def get_neighbors(self, doc_id: str, depth: int = 1):
        if doc_id not in self._graph:
            return []

        neighbors = set()
        current_level = {doc_id}

        for _ in range(depth):
            next_level = set()
            for node in current_level:
                successors = set(self._graph.successors(node))
                predecessors = set(self._graph.predecessors(node))
                next_level.update(successors | predecessors)

            neighbors.update(next_level)
            current_level = next_level

        neighbors.discard(doc_id)
        return list(neighbors)

    def persist(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)

        graph_data = nx.node_link_data(self._graph)
        graph_file = path / "graph.json"

        with open(graph_file, "w") as f:
            json.dump(graph_data, f, indent=2)

    def load(self, path: Path) -> None:
        graph_file = path / "graph.json"

        if not graph_file.exists():
            self._graph = nx.DiGraph()
            return

        with open(graph_file, "r") as f:
            graph_data = json.load(f)

        self._graph = nx.node_link_graph(graph_data, directed=True)
