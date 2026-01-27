"""Tag-based query expansion using graph relationships.

This module provides tag-graph intelligence to enhance search quality by:
1. Extracting tags from top results
2. Traversing tag graph to find related concepts
3. Expanding search with related tags
4. Fusing expanded results

Works transparently for both document and memory search.
"""

import logging

from src.indices.graph import GraphStore
from src.indices.vector import VectorIndex

logger = logging.getLogger(__name__)


def expand_query_with_tags(
    initial_results: list[dict],
    graph: GraphStore,
    vector: VectorIndex,
    top_k: int = 20,
    max_related_tags: int = 3,
    max_depth: int = 2,
) -> list[dict]:
    """Expand search results using tag graph relationships.

    Args:
        initial_results: Initial search results with chunk_id
        graph: Graph store with tag nodes and HAS_TAG edges
        vector: Vector index for retrieving chunks by tag
        top_k: Max results to return per tag
        max_related_tags: Max related tags to explore
        max_depth: Max depth for tag graph traversal

    Returns:
        Expanded list of search results
    """
    if not initial_results:
        return []

    # Extract tags from top N results
    top_results = initial_results[:5]  # Look at top 5 for tag extraction
    tags_found = set()

    for result in top_results:
        chunk_id = result.get("chunk_id", "")
        chunk_data = vector.get_chunk_by_id(chunk_id)
        if chunk_data:
            metadata = chunk_data.get("metadata", {})

            # Type guard: ensure metadata is a dict
            if not isinstance(metadata, dict):
                continue

            # Extract tags from metadata (works for both docs and memories)
            chunk_tags = metadata.get("tags", [])
            if not chunk_tags:
                # Try memory-specific tag field
                chunk_tags = metadata.get("memory_tags", [])

            if isinstance(chunk_tags, list):
                tags_found.update(chunk_tags)

    if not tags_found:
        logger.debug("No tags found in top results, skipping tag expansion")
        return initial_results

    # Find related tags via graph traversal
    related_tags = _find_related_tags(
        list(tags_found),
        graph,
        max_related_tags,
        max_depth,
    )

    if not related_tags:
        logger.debug("No related tags found in graph, skipping expansion")
        return initial_results

    # Find chunks with related tags
    expanded_chunks = []
    for tag in related_tags:
        tag_node_id = f"tag:{tag}"

        # Find nodes connected to this tag via HAS_TAG edges
        edges = graph.get_edges_from(tag_node_id)

        for edge in edges[:top_k]:
            target = edge.get("target", "")
            if target.startswith("memory:") or not target.startswith("ghost:"):
                # Get chunk IDs for this document/memory
                chunk_ids = vector.get_chunk_ids_for_document(target)

                for chunk_id in chunk_ids[:2]:  # Max 2 chunks per doc
                    chunk_data = vector.get_chunk_by_id(chunk_id)
                    if chunk_data:
                        expanded_chunks.append({
                            "chunk_id": chunk_id,
                            "doc_id": target,
                            "score": 0.5,  # Lower score for expanded results
                        })

    # Merge expanded results with initial results (dedupe by chunk_id)
    seen_chunk_ids = {r.get("chunk_id") for r in initial_results}
    unique_expanded = [
        r for r in expanded_chunks
        if r.get("chunk_id") not in seen_chunk_ids
    ]

    logger.debug(
        f"Tag expansion: found {len(tags_found)} tags, "
        f"{len(related_tags)} related tags, "
        f"added {len(unique_expanded)} new results"
    )

    return initial_results + unique_expanded


def _find_related_tags(
    seed_tags: list[str],
    graph: GraphStore,
    max_tags: int,
    max_depth: int,
) -> list[str]:
    """Find related tags via breadth-first graph traversal.

    Args:
        seed_tags: Initial tags to start from
        graph: Graph store
        max_tags: Maximum related tags to return
        max_depth: Maximum traversal depth

    Returns:
        List of related tag names (without 'tag:' prefix)
    """
    visited = set()
    queue = [(f"tag:{tag}", 0) for tag in seed_tags]
    related = []

    while queue and len(related) < max_tags:
        current_tag_id, depth = queue.pop(0)

        if current_tag_id in visited or depth >= max_depth:
            continue

        visited.add(current_tag_id)

        # Find edges from this tag to other tags
        edges = graph.get_edges_from(current_tag_id)

        for edge in edges:
            target = edge.get("target", "")

            # Only follow tag-to-tag edges
            if target.startswith("tag:") and target not in visited:
                edge_type = edge.get("edge_type", "")

                # Prefer RELATED_TO edges for tag relationships
                if edge_type in ("RELATED_TO", "related_to"):
                    tag_name = target.replace("tag:", "")
                    if tag_name not in seed_tags and tag_name not in related:
                        related.append(tag_name)
                        queue.append((target, depth + 1))

                        if len(related) >= max_tags:
                            break

    return related
