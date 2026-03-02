"""Unit tests for community-based scoring.

Tests cover:
- Community boosting calculation
- Community membership lookup
"""

from src.search.community import (
    compute_community_boost,
    get_community_members,
)


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
