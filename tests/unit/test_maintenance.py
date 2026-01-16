import pytest
from unittest.mock import MagicMock
import numpy as np
from src.memory.maintenance import MemoryGardener, MemoryCluster

class MockVectorIndex:
    def __init__(self, embeddings):
        self.embeddings = embeddings
        self.doc_ids = list(embeddings.keys())

    def get_document_ids(self):
        return self.doc_ids

    def get_chunk_by_id(self, doc_id):
        if doc_id in self.embeddings:
            return {
                "metadata": {"memory_type": "journal"},
                "file_path": f"/memories/{doc_id}.md",
                "content": f"Content for {doc_id}"
            }
        return None

    def get_embedding_for_chunk(self, doc_id):
        return self.embeddings.get(doc_id)

@pytest.fixture
def mock_manager():
    manager = MagicMock()
    # Create 3 vectors: A and B are close, C is far
    embeddings = {
        "mem1": [1.0, 0.0, 0.0],
        "mem2": [0.99, 0.1, 0.0],  # Very close to mem1
        "mem3": [0.0, 1.0, 0.0],   # Orthogonal to mem1
    }
    manager.vector = MockVectorIndex(embeddings)
    return manager

def test_find_clusters(mock_manager):
    gardener = MemoryGardener(mock_manager)
    
    # Threshold 0.9 should group mem1 and mem2
    clusters = gardener.find_clusters(threshold=0.9, min_cluster_size=2)
    
    assert len(clusters) == 1
    cluster = clusters[0]
    assert set(cluster.memory_ids) == {"mem1", "mem2"}
    assert cluster.score > 0.9

def test_find_clusters_no_matches(mock_manager):
    gardener = MemoryGardener(mock_manager)
    
    # Threshold 0.999 should find nothing (dot product of mem1, mem2 is < 1.0)
    # mem1.mem2 = 0.99 / (1 * sqrt(0.99^2 + 0.1^2)) ~= 0.995
    # Wait, 0.99*1 + 0*0.1 = 0.99. Norm(mem2) = sqrt(0.99^2 + 0.01) = sqrt(0.9901) ~= 0.995
    # Cosine sim = 0.99 / 0.995 ~= 0.994
    
    # If we set threshold 0.995, it might split them
    clusters = gardener.find_clusters(threshold=0.999, min_cluster_size=2)
    assert len(clusters) == 0

def test_filter_type():
    manager = MagicMock()
    embeddings = {"mem1": [1,0], "mem2": [1,0]}
    manager.vector = MockVectorIndex(embeddings)
    
    # Mock return values for get_chunk_by_id to have different types
    def get_chunk(doc_id):
        return {
            "metadata": {"memory_type": "journal" if doc_id == "mem1" else "plan"},
            "file_path": f"/{doc_id}",
            "content": "content"
        }
    manager.vector.get_chunk_by_id = get_chunk
    
    gardener = MemoryGardener(manager)
    clusters = gardener.find_clusters(threshold=0.5, filter_type="journal")
    # Should be 0 because mem2 is filtered out, leaving only mem1 (size < 2)
    assert len(clusters) == 0
