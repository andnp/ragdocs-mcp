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


async def _capture_stage_top_k(
    orchestrator: SearchOrchestrator,
    monkeypatch: pytest.MonkeyPatch,
    *,
    query_text: str,
    requested_top_k: int,
    top_n: int,
    project_filter: list[str] | None = None,
) -> dict[str, int]:
    observed: dict[str, int] = {}

    async def fake_search_vector(_query_text, top_k, _excluded_files, _docs_root):
        observed["vector"] = top_k
        return []

    async def fake_search_keyword(_query_text, top_k, _excluded_files, _docs_root):
        observed["keyword"] = top_k
        return []

    def fake_build_graph_chunk_candidates(
        _graph_neighbor_ids,
        _vector,
        top_k,
        *,
        excluded_chunk_ids,
    ):
        observed["graph"] = top_k
        assert isinstance(excluded_chunk_ids, set)
        return []

    monkeypatch.setattr(orchestrator, "_search_vector", fake_search_vector)
    monkeypatch.setattr(orchestrator, "_search_keyword", fake_search_keyword)
    monkeypatch.setattr(
        "src.search.orchestrator.build_graph_chunk_candidates",
        fake_build_graph_chunk_candidates,
    )

    await orchestrator.query(
        query_text,
        top_k=requested_top_k,
        top_n=top_n,
        project_filter=project_filter,
    )

    return observed


@pytest.mark.asyncio
async def test_small_factual_queries_contract_stage_top_k(
    orchestrator: SearchOrchestrator,
    monkeypatch: pytest.MonkeyPatch,
):
    observed = await _capture_stage_top_k(
        orchestrator,
        monkeypatch,
        query_text="get_user_by_id",
        requested_top_k=20,
        top_n=3,
    )

    assert observed == {"vector": 8, "keyword": 8, "graph": 8}


@pytest.mark.asyncio
async def test_exploratory_queries_keep_requested_stage_top_k(
    orchestrator: SearchOrchestrator,
    monkeypatch: pytest.MonkeyPatch,
):
    observed = await _capture_stage_top_k(
        orchestrator,
        monkeypatch,
        query_text="how does caching work",
        requested_top_k=20,
        top_n=3,
    )

    assert observed == {"vector": 20, "keyword": 20, "graph": 20}


@pytest.mark.asyncio
async def test_large_factual_queries_do_not_contract_stage_top_k(
    orchestrator: SearchOrchestrator,
    monkeypatch: pytest.MonkeyPatch,
):
    observed = await _capture_stage_top_k(
        orchestrator,
        monkeypatch,
        query_text="get_user_by_id",
        requested_top_k=20,
        top_n=6,
    )

    assert observed == {"vector": 20, "keyword": 20, "graph": 20}


@pytest.mark.asyncio
async def test_project_filtered_queries_keep_requested_stage_top_k(
    orchestrator: SearchOrchestrator,
    monkeypatch: pytest.MonkeyPatch,
):
    observed = await _capture_stage_top_k(
        orchestrator,
        monkeypatch,
        query_text="get_user_by_id",
        requested_top_k=20,
        top_n=3,
        project_filter=["docs-project"],
    )

    assert observed == {"vector": 20, "keyword": 20, "graph": 20}
