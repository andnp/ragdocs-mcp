import pytest

from src.config import ChunkingConfig, Config, IndexingConfig, LLMConfig, SearchConfig
from src.indexing.manager import IndexManager
from src.indices.graph import GraphStore
from src.indices.keyword import KeywordIndex
from src.indices.vector import VectorIndex
from tests.conftest import create_test_document


@pytest.fixture
def manager(tmp_path, shared_embedding_model):
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)

    config = Config(
        indexing=IndexingConfig(
            documents_path=str(docs_dir), index_path=str(tmp_path / ".index_data")
        ),
        search=SearchConfig(semantic_weight=1.0, keyword_weight=1.0, recency_bias=0.5),
        llm=LLMConfig(embedding_model="local"),
        chunking=ChunkingConfig(
            strategy="header_based",
            min_chunk_chars=200,
            max_chunk_chars=1500,
            overlap_chars=100,
        ),
    )

    vector = VectorIndex(embedding_model=shared_embedding_model)
    keyword = KeywordIndex()
    graph = GraphStore()
    return IndexManager(config, vector, keyword, graph)


def test_reindex_document_resolves_markdown_path(tmp_path, manager):
    """
    Verify reindex_document locates the markdown file and restores the document.

    Ensures:
    - doc_id resolves to a file under documents_path
    - reindex_document returns True when the file exists
    """
    docs_dir = tmp_path / "docs"
    doc_path = create_test_document(docs_dir, "reindex_me", "# Title\n\nBody")

    manager.index_document(doc_path)
    manager.remove_document("reindex_me")

    reindexed = manager.reindex_document("reindex_me", reason="test")

    assert reindexed is True
    assert "reindex_me" in manager.vector.get_document_ids()


def test_reindex_document_missing_file_returns_false(manager):
    """
    Verify reindex_document returns False when the source file is missing.

    Ensures:
    - missing doc_id does not raise
    - False is returned for missing files
    """
    reindexed = manager.reindex_document("missing_doc", reason="test")

    assert reindexed is False
