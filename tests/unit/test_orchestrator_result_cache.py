from pathlib import Path

import pytest

from src.config import ChunkingConfig, Config, IndexingConfig, LLMConfig, SearchConfig
from src.indexing.manager import IndexManager
from src.indices.graph import GraphStore
from src.indices.keyword import KeywordIndex
from src.indices.vector import VectorIndex
from src.search.orchestrator import SearchOrchestrator


@pytest.fixture
def config(tmp_path):
    docs_path = tmp_path / "docs"
    docs_path.mkdir()
    return Config(
        indexing=IndexingConfig(
            documents_path=str(docs_path),
            index_path=str(tmp_path / "indices"),
        ),
        search=SearchConfig(semantic_weight=1.0, keyword_weight=1.0),
        llm=LLMConfig(embedding_model="BAAI/bge-small-en-v1.5"),
        chunking=ChunkingConfig(),
    )


@pytest.fixture
def indices(shared_embedding_model):
    vector = VectorIndex(embedding_model=shared_embedding_model)
    keyword = KeywordIndex()
    graph = GraphStore()
    return vector, keyword, graph


@pytest.fixture
def manager(config, indices):
    vector, keyword, graph = indices
    return IndexManager(config, vector, keyword, graph)


@pytest.fixture
def orchestrator(config, indices, manager):
    vector, keyword, graph = indices
    return SearchOrchestrator(vector, keyword, graph, config, manager)


def _create_test_corpus(config: Config, manager: IndexManager) -> None:
    docs_path = Path(config.indexing.documents_path)
    (docs_path / "auth.md").write_text(
        "# Authentication\n\nAuthentication token validation and login flows."
    )
    (docs_path / "security.md").write_text(
        "# Security\n\nSecurity controls and authentication guidance."
    )

    manager.index_document(str(docs_path / "auth.md"))
    manager.index_document(str(docs_path / "security.md"))


def _install_search_counters(monkeypatch: pytest.MonkeyPatch, orchestrator: SearchOrchestrator) -> dict[str, int]:
    call_counts = {"vector": 0, "keyword": 0}
    original_search_vector = orchestrator._search_vector
    original_search_keyword = orchestrator._search_keyword

    async def counted_search_vector(query_text, top_k, excluded_files, docs_root):
        call_counts["vector"] += 1
        return await original_search_vector(query_text, top_k, excluded_files, docs_root)

    async def counted_search_keyword(query_text, top_k, excluded_files, docs_root):
        call_counts["keyword"] += 1
        return await original_search_keyword(query_text, top_k, excluded_files, docs_root)

    monkeypatch.setattr(orchestrator, "_search_vector", counted_search_vector)
    monkeypatch.setattr(orchestrator, "_search_keyword", counted_search_keyword)
    return call_counts


@pytest.mark.asyncio
async def test_repeated_query_uses_cached_results(
    config: Config,
    manager: IndexManager,
    orchestrator: SearchOrchestrator,
    monkeypatch: pytest.MonkeyPatch,
):
    """Given identical query inputs, when the query repeats, then search backends are not re-run."""
    _create_test_corpus(config, manager)
    call_counts = _install_search_counters(monkeypatch, orchestrator)

    first = await orchestrator.query("authentication", top_k=10, top_n=5)
    second = await orchestrator.query("authentication", top_k=10, top_n=5)

    assert call_counts == {"vector": 1, "keyword": 1}
    assert second == first


@pytest.mark.asyncio
async def test_query_cache_misses_when_query_shape_changes(
    config: Config,
    manager: IndexManager,
    orchestrator: SearchOrchestrator,
    monkeypatch: pytest.MonkeyPatch,
):
    """Given different excluded-files inputs, when the same text is queried, then the cache key changes."""
    _create_test_corpus(config, manager)
    call_counts = _install_search_counters(monkeypatch, orchestrator)

    await orchestrator.query("authentication", top_k=10, top_n=5)
    await orchestrator.query(
        "authentication",
        top_k=10,
        top_n=5,
        excluded_files={"security.md"},
    )

    assert call_counts == {"vector": 2, "keyword": 2}


@pytest.mark.asyncio
async def test_query_cache_invalidates_after_index_state_change(
    config: Config,
    manager: IndexManager,
    orchestrator: SearchOrchestrator,
    monkeypatch: pytest.MonkeyPatch,
):
    """Given a corpus mutation, when the same query repeats, then the cached result is invalidated."""
    _create_test_corpus(config, manager)
    call_counts = _install_search_counters(monkeypatch, orchestrator)

    await orchestrator.query("authentication", top_k=10, top_n=5)
    initial_state_version = manager.get_state_version()

    docs_path = Path(config.indexing.documents_path)
    extra_doc = docs_path / "auth_refresh.md"
    extra_doc.write_text(
        "# Authentication Refresh\n\nAuthentication refresh tokens and session renewal."
    )
    manager.index_document(str(extra_doc))

    await orchestrator.query("authentication", top_k=10, top_n=5)

    assert manager.get_state_version() > initial_state_version
    assert call_counts == {"vector": 2, "keyword": 2}