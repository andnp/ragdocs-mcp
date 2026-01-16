import pytest
from typing import TYPE_CHECKING, cast
from src.memory.maintenance import MemoryGardener

if TYPE_CHECKING:
    from src.memory.manager import MemoryIndexManager


class FakeVectorIndex:
    def __init__(self, embeddings: dict[str, list[float]], chunk_data: dict[str, dict]):
        self._embeddings = embeddings
        self._chunk_data = chunk_data
        self.doc_ids = list(embeddings.keys())

    def get_document_ids(self):
        return self.doc_ids

    def get_chunk_by_id(self, doc_id: str):
        return self._chunk_data.get(doc_id)

    def get_embedding_for_chunk(self, doc_id: str):
        return self._embeddings.get(doc_id)


class FakeMemoryManager:
    def __init__(self, vector: FakeVectorIndex):
        self.vector = vector


@pytest.fixture
def sample_manager():
    embeddings = {
        "mem1": [1.0, 0.0, 0.0],
        "mem2": [0.99, 0.1, 0.0],
        "mem3": [0.0, 1.0, 0.0],
    }
    chunk_data = {
        "mem1": {
            "metadata": {"memory_type": "journal"},
            "file_path": "/memories/mem1.md",
            "content": "Content for mem1"
        },
        "mem2": {
            "metadata": {"memory_type": "journal"},
            "file_path": "/memories/mem2.md",
            "content": "Content for mem2"
        },
        "mem3": {
            "metadata": {"memory_type": "journal"},
            "file_path": "/memories/mem3.md",
            "content": "Content for mem3"
        },
    }
    vector = FakeVectorIndex(embeddings, chunk_data)
    return cast("MemoryIndexManager", FakeMemoryManager(vector))


def test_find_clusters(sample_manager):
    """
    Verify that similar memories are grouped into clusters.

    Two memories with nearly identical embeddings (mem1, mem2) should form
    a single cluster when the similarity threshold is met.
    """
    gardener = MemoryGardener(sample_manager)

    clusters = gardener.find_clusters(threshold=0.9, min_cluster_size=2)

    assert len(clusters) == 1
    cluster = clusters[0]
    assert set(cluster.memory_ids) == {"mem1", "mem2"}
    assert cluster.score > 0.9


def test_find_clusters_no_matches(sample_manager):
    """
    Verify that a very high threshold results in no clusters.

    When the threshold is higher than the actual similarity between
    any pair of memories, no clusters should be formed.
    """
    gardener = MemoryGardener(sample_manager)

    clusters = gardener.find_clusters(threshold=0.999, min_cluster_size=2)
    assert len(clusters) == 0


def test_filter_type():
    """
    Verify that type filtering excludes memories of non-matching types.

    When filtering for 'journal' type only, memories of other types
    should be excluded, potentially resulting in smaller or no clusters.
    """
    embeddings = {"mem1": [1.0, 0.0], "mem2": [1.0, 0.0]}
    chunk_data = {
        "mem1": {
            "metadata": {"memory_type": "journal"},
            "file_path": "/mem1.md",
            "content": "content"
        },
        "mem2": {
            "metadata": {"memory_type": "plan"},
            "file_path": "/mem2.md",
            "content": "content"
        },
    }
    vector = FakeVectorIndex(embeddings, chunk_data)
    manager = cast("MemoryIndexManager", FakeMemoryManager(vector))

    gardener = MemoryGardener(manager)
    clusters = gardener.find_clusters(threshold=0.5, filter_type="journal")
    assert len(clusters) == 0
