import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import networkx as nx
import numpy as np
from src.memory.manager import MemoryIndexManager

if TYPE_CHECKING:
    from src.context import ApplicationContext

logger = logging.getLogger(__name__)


@dataclass
class MemoryCluster:
    cluster_id: int
    memory_ids: list[str]
    representative_title: str  # derived from the first memory or a common tag
    reason: str
    score: float


class MemoryGardener:
    def __init__(self, manager: MemoryIndexManager):
        self.manager = manager

    def find_clusters(
        self,
        threshold: float = 0.85,
        min_cluster_size: int = 2,
        limit: int = 5,
        filter_type: str | None = None
    ) -> list[MemoryCluster]:
        """
        Find clusters of related memories that are candidates for merging.
        Uses vector similarity + graph connected components.
        """
        # 1. Get all valid memory IDs
        doc_ids = self.manager.vector.get_document_ids()
        valid_ids = []
        embeddings = []
        
        # Filter and collect embeddings
        for doc_id in doc_ids:
            # Check type filter if requested
            # Note: doc_id in vector index is chunk_id or memory_id. 
            # We assume 1:1 mapping for memories or we aggregate.
            # For simplicity, we assume we want to cluster full memory documents.
            
            # We need to peek at metadata to check type
            # The VectorIndex.get_chunk_by_id returns metadata
            chunk_data = self.manager.vector.get_chunk_by_id(doc_id)
            if not chunk_data:
                continue
                
            metadata = chunk_data.get("metadata", {})
            if filter_type and metadata.get("memory_type") != filter_type:
                continue
                
            # Skip if it's already a specialized type that shouldn't be merged?
            # Maybe keep everything.

            emb = self.manager.vector.get_embedding_for_chunk(doc_id)
            if emb:
                valid_ids.append(doc_id)
                embeddings.append(emb)

        if not valid_ids:
            return []

        # 2. Build Similarity Graph
        # We use a simple O(N^2) comparison since memory bank is expected to be < 10k items
        # For larger banks, we would use FAISS IVFFlat or HNSW for approx NN, 
        # but here we want precise clusters.
        
        matrix = np.array(embeddings)
        # Normalize
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        # Avoid division by zero
        norms[norms == 0] = 1.0
        normalized = matrix / norms
        
        # Compute similarity matrix
        sim_matrix = np.dot(normalized, normalized.T)
        
        # 3. Find Connected Components
        G = nx.Graph()
        G.add_nodes_from(valid_ids)
        
        rows, cols = np.where(sim_matrix > threshold)
        for r, c in zip(rows, cols):
            if r < c:  # upper triangle only, avoid self-loops
                G.add_edge(valid_ids[r], valid_ids[c], weight=float(sim_matrix[r, c]))
                
        components = list(nx.connected_components(G))
        
        # 4. Format Results
        clusters = []
        for i, comp in enumerate(components):
            if len(comp) < min_cluster_size:
                continue
                
            members = list(comp)
            
            # Get details for the first member to use as representative
            first_mem = self.manager.vector.get_chunk_by_id(members[0])
            title = "Untitled Cluster"
            if first_mem:
                # Try to get a title from content or path
                path = first_mem.get("file_path", "")
                title = f"Cluster related to {Path(path).stem}"
            
            # Calculate average similarity score of the cluster
            subgraph = G.subgraph(members)
            if subgraph.number_of_edges() > 0:
                avg_score = sum(d['weight'] for u, v, d in subgraph.edges(data=True)) / subgraph.number_of_edges()
            else:
                avg_score = 1.0

            clusters.append(MemoryCluster(
                cluster_id=i,
                memory_ids=members,
                representative_title=title,
                reason=f"High vector similarity (> {threshold})",
                score=avg_score
            ))
            
        # Sort by score (tightest clusters first)
        clusters.sort(key=lambda x: x.score, reverse=True)
        
        return clusters[:limit]


from pathlib import Path

async def suggest_memory_merges(
    ctx: "ApplicationContext",
    threshold: float = 0.85,
    limit: int = 5,
    filter_type: str | None = "journal"
) -> list[dict]:
    """
    Suggests groups of memories that could be merged based on content similarity.
    This is the 'Scout' for the memory maintenance workflow.
    """
    if ctx.memory_manager is None:
        return [{"error": "Memory system is not enabled"}]
        
    gardener = MemoryGardener(ctx.memory_manager)
    
    try:
        clusters = gardener.find_clusters(
            threshold=threshold,
            limit=limit,
            filter_type=filter_type
        )
        
        results = []
        for c in clusters:
            # Fetch brief content for each memory to help user decide
            memories = []
            for mid in c.memory_ids:
                # mid is likely the chunk_id. Let's try to map back to filename if possible
                # or just use the chunk content
                chunk = ctx.memory_manager.vector.get_chunk_by_id(mid)
                if chunk:
                    memories.append({
                        "id": mid,
                        "file_path": chunk.get("file_path"),
                        "preview": chunk.get("content", "")[:100] + "..."
                    })
            
            results.append({
                "cluster_id": c.cluster_id,
                "score": f"{c.score:.2f}",
                "reason": c.reason,
                "memory_count": len(c.memory_ids),
                "memories": memories
            })
            
        return results
        
    except Exception as e:
        logger.error(f"Failed to suggest merges: {e}", exc_info=True)
        return [{"error": str(e)}]
