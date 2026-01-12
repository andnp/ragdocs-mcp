"""
Unit tests for community detection and community-based scoring.

Tests cover:
- Community detection algorithms (Louvain, Label Propagation)
- Empty and simple graph handling
- Community index lookup and membership
- Community boosting calculation
- Fallback behavior when algorithms fail
"""

import pytest
import networkx as nx

from src.search.community import (
    LabelPropagationDetector,
    LouvainDetector,
    get_community_detector,
    compute_community_boost,
    get_community_members,
)


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def empty_graph():
    return nx.DiGraph()


@pytest.fixture
def single_node_graph():
    g = nx.DiGraph()
    g.add_node("doc1")
    return g


@pytest.fixture
def disconnected_graph():
    """Graph with two separate connected components."""
    g = nx.DiGraph()
    g.add_edge("a1", "a2")
    g.add_edge("a2", "a3")
    g.add_edge("b1", "b2")
    g.add_edge("b2", "b3")
    return g


@pytest.fixture
def fully_connected_graph():
    """Graph where all nodes are interconnected (single community)."""
    g = nx.DiGraph()
    nodes = ["doc1", "doc2", "doc3", "doc4"]
    for i, n1 in enumerate(nodes):
        for n2 in nodes[i + 1:]:
            g.add_edge(n1, n2)
            g.add_edge(n2, n1)
    return g


@pytest.fixture
def clustered_graph():
    """Graph with two distinct clusters connected by a bridge."""
    g = nx.DiGraph()
    g.add_edge("a1", "a2")
    g.add_edge("a2", "a3")
    g.add_edge("a1", "a3")
    g.add_edge("b1", "b2")
    g.add_edge("b2", "b3")
    g.add_edge("b1", "b3")
    g.add_edge("a3", "b1")
    return g


# ============================================================================
# Community Detector Factory Tests
# ============================================================================


class TestGetCommunityDetector:
    """Tests for the community detector factory function."""

    def test_default_returns_louvain(self):
        """
        Default algorithm should be Louvain (higher quality communities).
        """
        detector = get_community_detector()
        assert isinstance(detector, LouvainDetector)

    def test_explicit_louvain_selection(self):
        """
        Explicitly requesting 'louvain' returns LouvainDetector.
        """
        detector = get_community_detector("louvain")
        assert isinstance(detector, LouvainDetector)

    def test_label_propagation_selection(self):
        """
        Requesting 'label_propagation' returns LabelPropagationDetector.
        """
        detector = get_community_detector("label_propagation")
        assert isinstance(detector, LabelPropagationDetector)

    def test_unknown_algorithm_returns_louvain(self):
        """
        Unknown algorithm names should fall back to Louvain.
        """
        detector = get_community_detector("unknown_algo")
        assert isinstance(detector, LouvainDetector)


# ============================================================================
# Label Propagation Detector Tests
# ============================================================================


class TestLabelPropagationDetector:
    """Tests for the LabelPropagationDetector community detection."""

    def test_empty_graph_returns_empty_dict(self, empty_graph):
        """
        Empty graph should return empty community mapping.
        No nodes means no communities to detect.
        """
        detector = LabelPropagationDetector()
        result = detector.detect(empty_graph)
        assert result == {}

    def test_single_node_graph(self, single_node_graph):
        """
        Single node graph should assign that node to a community.
        """
        detector = LabelPropagationDetector()
        result = detector.detect(single_node_graph)
        assert "doc1" in result
        assert isinstance(result["doc1"], int)

    def test_disconnected_components_get_different_communities(self, disconnected_graph):
        """
        Disconnected components should be assigned different community IDs.
        """
        detector = LabelPropagationDetector()
        result = detector.detect(disconnected_graph)

        cluster_a = {result["a1"], result["a2"], result["a3"]}
        cluster_b = {result["b1"], result["b2"], result["b3"]}

        assert result["a1"] == result["a2"] == result["a3"]
        assert result["b1"] == result["b2"] == result["b3"]
        assert cluster_a != cluster_b

    def test_fully_connected_forms_single_community(self, fully_connected_graph):
        """
        Fully connected graph should form a single community.
        """
        detector = LabelPropagationDetector()
        result = detector.detect(fully_connected_graph)

        community_ids = set(result.values())
        assert len(community_ids) == 1

    def test_returns_string_keys(self, disconnected_graph):
        """
        Community mapping keys should be strings (doc IDs).
        """
        detector = LabelPropagationDetector()
        result = detector.detect(disconnected_graph)

        for key in result:
            assert isinstance(key, str)


# ============================================================================
# Louvain Detector Tests
# ============================================================================


class TestLouvainDetector:
    """Tests for the LouvainDetector community detection."""

    def test_empty_graph_returns_empty_dict(self, empty_graph):
        """
        Empty graph should return empty community mapping.
        """
        detector = LouvainDetector()
        result = detector.detect(empty_graph)
        assert result == {}

    def test_single_node_graph(self, single_node_graph):
        """
        Single node graph should assign that node to a community.
        """
        detector = LouvainDetector()
        result = detector.detect(single_node_graph)
        assert "doc1" in result
        assert isinstance(result["doc1"], int)

    def test_disconnected_components(self, disconnected_graph):
        """
        Disconnected components should be in different communities.
        """
        detector = LouvainDetector()
        result = detector.detect(disconnected_graph)

        assert result["a1"] == result["a2"] == result["a3"]
        assert result["b1"] == result["b2"] == result["b3"]
        assert result["a1"] != result["b1"]

    def test_resolution_parameter_affects_granularity(self, clustered_graph):
        """
        Higher resolution should produce more communities (finer granularity).
        Lower resolution should produce fewer communities (coarser).
        """
        low_res = LouvainDetector(resolution=0.5)
        high_res = LouvainDetector(resolution=2.0)

        result_low = low_res.detect(clustered_graph)
        result_high = high_res.detect(clustered_graph)

        communities_low = len(set(result_low.values()))
        communities_high = len(set(result_high.values()))

        assert communities_high >= communities_low


# ============================================================================
# Community Boost Calculation Tests
# ============================================================================


class TestComputeCommunityBoost:
    """Tests for community-based score boosting."""

    def test_same_community_gets_boost(self):
        """
        Documents in the same community as seed should receive boost factor.
        """
        communities = {"doc1": 0, "doc2": 0, "doc3": 1}
        doc_ids = ["doc1", "doc2", "doc3"]
        seed_doc_ids = {"doc1"}
        boost_factor = 1.2

        boosts = compute_community_boost(doc_ids, communities, seed_doc_ids, boost_factor)

        assert boosts["doc1"] == 1.2
        assert boosts["doc2"] == 1.2
        assert boosts["doc3"] == 1.0

    def test_different_community_no_boost(self):
        """
        Documents in different communities from seeds should not receive boost.
        """
        communities = {"doc1": 0, "doc2": 1}
        doc_ids = ["doc1", "doc2"]
        seed_doc_ids = {"doc1"}
        boost_factor = 1.5

        boosts = compute_community_boost(doc_ids, communities, seed_doc_ids, boost_factor)

        assert boosts["doc1"] == 1.5
        assert boosts["doc2"] == 1.0

    def test_multiple_seed_communities(self):
        """
        Documents in any seed community should receive boost.
        """
        communities = {"doc1": 0, "doc2": 1, "doc3": 2, "doc4": 0, "doc5": 1}
        doc_ids = ["doc1", "doc2", "doc3", "doc4", "doc5"]
        seed_doc_ids = {"doc1", "doc2"}
        boost_factor = 1.1

        boosts = compute_community_boost(doc_ids, communities, seed_doc_ids, boost_factor)

        assert boosts["doc1"] == 1.1
        assert boosts["doc2"] == 1.1
        assert boosts["doc3"] == 1.0
        assert boosts["doc4"] == 1.1
        assert boosts["doc5"] == 1.1

    def test_empty_seed_docs_no_boost(self):
        """
        Empty seed set should result in no boosts applied.
        """
        communities = {"doc1": 0, "doc2": 0}
        doc_ids = ["doc1", "doc2"]
        seed_doc_ids = set()
        boost_factor = 1.5

        boosts = compute_community_boost(doc_ids, communities, seed_doc_ids, boost_factor)

        assert boosts["doc1"] == 1.0
        assert boosts["doc2"] == 1.0

    def test_missing_community_assignment(self):
        """
        Documents without community assignment should not receive boost.
        """
        communities = {"doc1": 0}
        doc_ids = ["doc1", "doc2"]
        seed_doc_ids = {"doc1"}
        boost_factor = 1.3

        boosts = compute_community_boost(doc_ids, communities, seed_doc_ids, boost_factor)

        assert boosts["doc1"] == 1.3
        assert boosts["doc2"] == 1.0

    def test_seed_not_in_communities(self):
        """
        Seed documents without community assignment should be ignored gracefully.
        """
        communities = {"doc1": 0, "doc2": 0}
        doc_ids = ["doc1", "doc2"]
        seed_doc_ids = {"unknown_seed"}
        boost_factor = 1.2

        boosts = compute_community_boost(doc_ids, communities, seed_doc_ids, boost_factor)

        assert boosts["doc1"] == 1.0
        assert boosts["doc2"] == 1.0

    def test_default_boost_factor(self):
        """
        Default boost factor should be applied correctly.
        """
        communities = {"doc1": 0, "doc2": 0}
        doc_ids = ["doc1", "doc2"]
        seed_doc_ids = {"doc1"}

        boosts = compute_community_boost(doc_ids, communities, seed_doc_ids)

        assert boosts["doc1"] == 1.1
        assert boosts["doc2"] == 1.1


# ============================================================================
# Community Members Lookup Tests
# ============================================================================


class TestGetCommunityMembers:
    """Tests for retrieving community membership lists."""

    def test_get_members_of_existing_community(self):
        """
        Should return all documents assigned to the specified community.
        """
        communities = {"doc1": 0, "doc2": 0, "doc3": 1, "doc4": 0}
        members = get_community_members(communities, 0)

        assert set(members) == {"doc1", "doc2", "doc4"}

    def test_get_members_of_nonexistent_community(self):
        """
        Non-existent community ID should return empty list.
        """
        communities = {"doc1": 0, "doc2": 1}
        members = get_community_members(communities, 99)

        assert members == []

    def test_empty_communities_dict(self):
        """
        Empty communities dict should return empty list for any query.
        """
        communities = {}
        members = get_community_members(communities, 0)

        assert members == []

    def test_single_member_community(self):
        """
        Community with single member should return list with one doc.
        """
        communities = {"doc1": 0, "doc2": 1}
        members = get_community_members(communities, 1)

        assert members == ["doc2"]
