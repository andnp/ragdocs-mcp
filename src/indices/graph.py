import json
from pathlib import Path
from threading import Lock
from typing import Any

import networkx as nx

from src.indices.protocol import SearchResult


class GraphStore:
    def __init__(self):
        self._graph: nx.DiGraph = nx.DiGraph()
        self._lock = Lock()

    def add_node(self, doc_id: str, metadata: dict) -> None:
        with self._lock:
            self._graph.add_node(doc_id, **metadata)

    def add_edge(self, source: str, target: str, edge_type: str) -> None:
        with self._lock:
            self._graph.add_edge(source, target, edge_type=edge_type)

    def remove_node(self, doc_id: str) -> None:
        with self._lock:
            if doc_id in self._graph:
                self._graph.remove_node(doc_id)

    def get_neighbors(self, doc_id: str, depth: int = 1):
        with self._lock:
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
        with self._lock:
            path.mkdir(parents=True, exist_ok=True)

            graph_data = nx.node_link_data(self._graph)
            graph_file = path / "graph.json"

            with open(graph_file, "w") as f:
                json.dump(graph_data, f, indent=2)

    def load(self, path: Path) -> None:
        with self._lock:
            graph_file = path / "graph.json"

            if not graph_file.exists():
                self._graph = nx.DiGraph()
                return

            with open(graph_file, "r") as f:
                graph_data = json.load(f)

            self._graph = nx.node_link_graph(graph_data, directed=True)

    # IndexProtocol methods

    def search(self, query: str, limit: int = 10) -> list[SearchResult]:
        with self._lock:
            results = []
            for doc_id in self._graph.nodes():
                metadata = dict(self._graph.nodes[doc_id])
                content = metadata.get("content", "")
                if query.lower() in str(doc_id).lower() or query.lower() in str(content).lower():
                    results.append(SearchResult(doc_id=doc_id, score=1.0, metadata=metadata))
                    if len(results) >= limit:
                        break
            return results

    def add_document(self, doc_id: str, content: str, metadata: dict[str, Any]) -> None:
        with self._lock:
            self._graph.add_node(doc_id, content=content, **metadata)

    def remove_document(self, doc_id: str) -> None:
        self.remove_node(doc_id)

    def clear(self) -> None:
        with self._lock:
            self._graph = nx.DiGraph()

    def save(self, path: Path) -> None:
        self.persist(path)

    def __len__(self) -> int:
        with self._lock:
            return self._graph.number_of_nodes()
