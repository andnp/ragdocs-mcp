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


class _ReadyVectorStub:
    def __init__(self, *, index_loaded: bool, model_loaded: bool):
        self._index_loaded = index_loaded
        self._model_loaded = model_loaded

    def is_ready(self) -> bool:
        return self._index_loaded

    def model_ready(self) -> bool:
        return self._model_loaded


def test_is_ready_returns_true_for_loaded_index_without_model_prewarm(manager):
    """
    Verify a loaded persisted index is queryable before embedding warm-up.

    Ensures:
    - readiness depends on loaded index state
    - lazy embedding model initialization does not block queries
    """
    manager.vector = _ReadyVectorStub(index_loaded=True, model_loaded=False)

    assert manager.is_ready() is True


def test_is_ready_returns_false_when_no_index_is_loaded(manager):
    """
    Verify readiness stays false when no persisted index snapshot is loaded.

    Ensures:
    - true cold starts remain blocked
    - model readiness alone cannot make the manager queryable
    """
    manager.vector = _ReadyVectorStub(index_loaded=False, model_loaded=True)

    assert manager.is_ready() is False
