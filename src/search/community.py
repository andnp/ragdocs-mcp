import logging

logger = logging.getLogger(__name__)


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
