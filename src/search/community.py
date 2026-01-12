import logging
from typing import Protocol

import networkx as nx

logger = logging.getLogger(__name__)


class CommunityDetector(Protocol):
    def detect(self, graph: nx.DiGraph) -> dict[str, int]:
        ...


class LabelPropagationDetector:
    def __init__(self, max_iterations: int = 100):
        self._max_iterations = max_iterations

    def detect(self, graph: nx.DiGraph):
        if graph.number_of_nodes() == 0:
            return {}

        undirected = graph.to_undirected()

        try:
            communities = nx.community.label_propagation_communities(undirected)
            result: dict[str, int] = {}

            for community_id, community in enumerate(communities):
                for node in community:
                    result[str(node)] = community_id

            return result
        except Exception as e:
            logger.warning(f"Label propagation failed: {e}, falling back to connected components")
            return self._fallback_components(undirected)

    def _fallback_components(self, graph: nx.Graph):
        result: dict[str, int] = {}
        for community_id, component in enumerate(nx.connected_components(graph)):
            for node in component:
                result[str(node)] = community_id
        return result


class LouvainDetector:
    def __init__(self, resolution: float = 1.0):
        self._resolution = resolution

    def detect(self, graph: nx.DiGraph):
        if graph.number_of_nodes() == 0:
            return {}

        undirected = graph.to_undirected()

        try:
            partition = nx.community.louvain_communities(
                undirected,
                resolution=self._resolution,
            )
            result: dict[str, int] = {}

            for community_id, community in enumerate(partition):
                for node in community:
                    result[str(node)] = community_id

            return result
        except Exception as e:
            logger.warning(f"Louvain detection failed: {e}, falling back to label propagation")
            fallback = LabelPropagationDetector()
            return fallback.detect(graph)


def get_community_detector(algorithm: str = "louvain"):
    if algorithm == "label_propagation":
        return LabelPropagationDetector()
    return LouvainDetector()


def compute_community_boost(
    doc_ids: list[str],
    communities: dict[str, int],
    seed_doc_ids: set[str],
    boost_factor: float = 1.1,
):
    seed_communities: set[int] = set()
    for doc_id in seed_doc_ids:
        if doc_id in communities:
            seed_communities.add(communities[doc_id])

    boosts: dict[str, float] = {}
    for doc_id in doc_ids:
        community_id = communities.get(doc_id)
        if community_id is not None and community_id in seed_communities:
            boosts[doc_id] = boost_factor
        else:
            boosts[doc_id] = 1.0

    return boosts


def get_community_members(communities: dict[str, int], community_id: int):
    return [doc_id for doc_id, cid in communities.items() if cid == community_id]
