import json
import logging
from pathlib import Path
from threading import Lock
from typing import Any

import networkx as nx

from src.indices.protocol import SearchResult
from src.search.community import get_community_detector, compute_community_boost, get_community_members

logger = logging.getLogger(__name__)


class GraphStore:
    def __init__(self):
        self._graph: nx.DiGraph = nx.DiGraph()
        self._lock = Lock()
        self._communities: dict[str, int] = {}
        self._community_detection_enabled = True

    def add_node(self, doc_id: str, metadata: dict) -> None:
        with self._lock:
            self._graph.add_node(doc_id, **metadata)

    def add_edge(
        self,
        source: str,
        target: str,
        edge_type: str,
        edge_context: str = "",
    ) -> None:
        with self._lock:
            self._graph.add_edge(
                source,
                target,
                edge_type=edge_type,
                edge_context=edge_context,
            )

    def get_edges_to(self, target: str) -> list[dict[str, str]]:
        with self._lock:
            if target not in self._graph:
                return []

            edges = []
            for source in self._graph.predecessors(target):
                edge_data = self._graph.edges[source, target]
                edges.append({
                    "source": source,
                    "target": target,
                    "edge_type": edge_data.get("edge_type", "related_to"),
                    "edge_context": edge_data.get("edge_context", ""),
                })

            return edges

    def get_edges_from(self, source: str) -> list[dict[str, str]]:
        """Get all edges originating from the given node."""
        with self._lock:
            if source not in self._graph:
                return []

            edges = []
            for target in self._graph.successors(source):
                edge_data = self._graph.edges[source, target]
                edges.append({
                    "source": source,
                    "target": target,
                    "edge_type": edge_data.get("edge_type", "related_to"),
                    "edge_context": edge_data.get("edge_context", ""),
                })

            return edges

    def has_node(self, doc_id: str) -> bool:
        with self._lock:
            return doc_id in self._graph

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

    def detect_communities(self, algorithm: str = "louvain") -> dict[str, int]:
        with self._lock:
            if self._graph.number_of_nodes() == 0:
                self._communities = {}
                return {}

            detector = get_community_detector(algorithm)
            self._communities = detector.detect(self._graph)
            logger.info(f"Detected {len(set(self._communities.values()))} communities across {len(self._communities)} nodes")
            return self._communities

    def get_community(self, doc_id: str) -> int | None:
        with self._lock:
            return self._communities.get(doc_id)

    def get_community_members(self, community_id: int) -> list[str]:
        with self._lock:
            return get_community_members(self._communities, community_id)

    def boost_by_community(
        self,
        doc_ids: list[str],
        seed_doc_ids: set[str],
        boost_factor: float = 1.1,
    ) -> dict[str, float]:
        with self._lock:
            return compute_community_boost(
                doc_ids, self._communities, seed_doc_ids, boost_factor
            )

    def set_community_detection_enabled(self, enabled: bool) -> None:
        self._community_detection_enabled = enabled

    def persist(self, path: Path) -> None:
        with self._lock:
            path.mkdir(parents=True, exist_ok=True)

            graph_data = nx.node_link_data(self._graph)
            graph_file = path / "graph.json"

            with open(graph_file, "w") as f:
                json.dump(graph_data, f, indent=2)

            if self._community_detection_enabled and self._graph.number_of_nodes() > 0:
                detector = get_community_detector("louvain")
                self._communities = detector.detect(self._graph)
                logger.info(f"Persisting {len(set(self._communities.values()))} communities")

            if self._communities:
                communities_file = path / "communities.json"
                with open(communities_file, "w") as f:
                    json.dump(self._communities, f, indent=2)

    def load(self, path: Path) -> None:
        with self._lock:
            graph_file = path / "graph.json"

            if not graph_file.exists():
                self._graph = nx.DiGraph()
                self._communities = {}
                return

            try:
                with open(graph_file, "r") as f:
                    graph_data = json.load(f)
                self._graph = nx.node_link_graph(graph_data, directed=True)
            except (json.JSONDecodeError, TypeError, KeyError, AttributeError) as e:
                logger.warning(
                    f"Graph index corruption detected (malformed graph.json): {e}. "
                    "Reinitializing graph.",
                    exc_info=True,
                )
                self._reinitialize_after_corruption()
                return

            communities_file = path / "communities.json"
            if communities_file.exists():
                try:
                    with open(communities_file, "r") as f:
                        self._communities = json.load(f)
                    logger.info(f"Loaded {len(set(self._communities.values()))} communities")
                except json.JSONDecodeError as e:
                    logger.warning(
                        f"Communities file corruption detected (malformed communities.json): {e}. "
                        "Continuing with empty communities.",
                        exc_info=True,
                    )
                    self._communities = {}
            else:
                self._communities = {}

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
            self._communities = {}

    def _reinitialize_after_corruption(self) -> None:
        self._graph = nx.DiGraph()
        self._communities = {}

    def save(self, path: Path) -> None:
        self.persist(path)

    def __len__(self) -> int:
        with self._lock:
            return self._graph.number_of_nodes()
