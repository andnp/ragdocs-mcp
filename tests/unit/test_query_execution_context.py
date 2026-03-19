from pathlib import Path

import pytest

from src.config import ChunkingConfig, Config, IndexingConfig, SearchConfig
from src.indices.graph import GraphStore
from src.indices.keyword import KeywordIndex
from src.indices.vector import VectorIndex
from src.models import CompressionStats
from src.search.classifier import classify_query
from src.search.orchestrator import SearchOrchestrator


def _orchestrator(*, detected_project: str | None = None) -> SearchOrchestrator:
    config = Config(
        indexing=IndexingConfig(
            documents_path="/tmp/docs",
            index_path="/tmp/index",
        ),
        search=SearchConfig(),
        chunking=ChunkingConfig(),
        detected_project=detected_project,
    )
    return SearchOrchestrator(
        VectorIndex(embedding_model_name="BAAI/bge-small-en-v1.5"),
        KeywordIndex(),
        GraphStore(),
        config,
        None,
        documents_path=Path(config.indexing.documents_path),
    )


def _result(chunk_id: str, score: float = 0.5) -> dict[str, object]:
    doc_id = chunk_id.split("_chunk_", 1)[0]
    return {"chunk_id": chunk_id, "doc_id": doc_id, "score": score}


def test_query_execution_context_reuses_metadata_for_uplift_filter_parent_and_materialize(
    monkeypatch,
) -> None:
    orchestrator = _orchestrator(detected_project="project-a")
    query_context = orchestrator._create_query_execution_context()

    chunk_data = {
        "child": {
            "chunk_id": "child",
            "doc_id": "doc",
            "header_path": "Section",
            "file_path": "/tmp/docs/doc.md",
            "content": "Child content",
            "metadata": {"project_id": "project-a", "parent_chunk_id": "parent"},
        },
        "parent": {
            "chunk_id": "parent",
            "doc_id": "doc",
            "header_path": "Parent",
            "file_path": "/tmp/docs/doc.md",
            "content": "Parent content",
            "metadata": {"project_id": "project-a"},
        },
    }
    lookup_counts = {"child": 0, "parent": 0}

    def fake_get_chunk_by_id(chunk_id: str):
        lookup_counts[chunk_id] = lookup_counts.get(chunk_id, 0) + 1
        return chunk_data.get(chunk_id)

    parent_content_calls = {"count": 0}

    def fake_get_parent_content(parent_chunk_id: str) -> str | None:
        parent_content_calls["count"] += 1
        return chunk_data[parent_chunk_id]["content"]

    monkeypatch.setattr(orchestrator._vector, "get_chunk_by_id", fake_get_chunk_by_id)
    monkeypatch.setattr(
        orchestrator._vector,
        "get_parent_content",
        fake_get_parent_content,
    )

    fused = [("child", 0.4)]
    boosted = orchestrator._apply_project_uplift(
        fused,
        query_context=query_context,
        project_context="project-a",
    )
    filtered = orchestrator._apply_project_filter(
        boosted,
        query_context=query_context,
        project_filter=["project-a"],
    )
    expanded = orchestrator._expand_to_parents(filtered, query_context=query_context)
    chunk_results = orchestrator._materialize_chunk_results(
        expanded,
        query_context=query_context,
    )

    assert [result.chunk_id for result in chunk_results] == ["parent"]
    assert chunk_results[0].parent_content is None
    assert lookup_counts == {"child": 1, "parent": 1}
    assert parent_content_calls["count"] == 0

    stats = query_context.stats
    assert stats.metadata_lookups == 2
    assert stats.metadata_cache_hits >= 2
    assert stats.parent_lookups == 1
    assert stats.parent_cache_hits == 0


def test_query_execution_context_caches_content_embedding_and_parent_content(
    monkeypatch,
) -> None:
    orchestrator = _orchestrator()
    query_context = orchestrator._create_query_execution_context()

    chunk_data = {
        "child": {
            "chunk_id": "child",
            "doc_id": "doc",
            "header_path": "Section",
            "file_path": "/tmp/docs/doc.md",
            "content": "Child content",
            "metadata": {"parent_chunk_id": "parent"},
        },
        "parent": {
            "chunk_id": "parent",
            "doc_id": "doc",
            "header_path": "Parent",
            "file_path": "/tmp/docs/doc.md",
            "content": "Parent content",
            "metadata": {},
        },
    }
    call_counts = {"chunk": 0, "embedding": 0, "parent_content": 0}

    def fake_get_chunk_by_id(chunk_id: str):
        call_counts["chunk"] += 1
        return chunk_data.get(chunk_id)

    def fake_get_embedding_for_chunk(chunk_id: str) -> list[float] | None:
        call_counts["embedding"] += 1
        return [0.25, 0.75] if chunk_id == "child" else None

    def fake_get_parent_content(parent_chunk_id: str) -> str | None:
        call_counts["parent_content"] += 1
        return chunk_data[parent_chunk_id]["content"]

    monkeypatch.setattr(orchestrator._vector, "get_chunk_by_id", fake_get_chunk_by_id)
    monkeypatch.setattr(
        orchestrator._vector,
        "get_embedding_for_chunk",
        fake_get_embedding_for_chunk,
    )
    monkeypatch.setattr(
        orchestrator._vector,
        "get_parent_content",
        fake_get_parent_content,
    )

    assert query_context.get_chunk_content("child") == "Child content"
    assert query_context.get_chunk_content("child") == "Child content"
    assert query_context.get_chunk_embedding("child") == [0.25, 0.75]
    assert query_context.get_chunk_embedding("child") == [0.25, 0.75]

    hydrated = query_context.hydrate_chunk_result("child", 0.8)
    hydrated_again = query_context.hydrate_chunk_result("child", 0.8)

    assert hydrated is not None
    assert hydrated_again is not None
    assert hydrated.parent_content == "Parent content"
    assert hydrated_again.parent_content == "Parent content"
    assert call_counts == {"chunk": 1, "embedding": 1, "parent_content": 1}

    stats = query_context.stats
    assert stats.content_lookups == 1
    assert stats.content_cache_hits == 1
    assert stats.embedding_fetches == 1
    assert stats.embedding_cache_hits == 1
    assert stats.parent_lookups == 2
    assert stats.parent_cache_hits == 1


@pytest.mark.asyncio
async def test_query_uses_single_execution_context_and_records_stats(monkeypatch) -> None:
    orchestrator = _orchestrator(detected_project="project-a")
    chunk_id = "doc-a_chunk_0"

    chunk_data = {
        chunk_id: {
            "chunk_id": chunk_id,
            "doc_id": "doc-a",
            "header_path": "Section",
            "file_path": "/tmp/docs/doc-a.md",
            "content": "Alpha content",
            "metadata": {"project_id": "project-a"},
        }
    }
    call_counts = {"chunk": 0, "embedding": 0}

    def fake_get_chunk_by_id(chunk_id: str):
        call_counts["chunk"] += 1
        return chunk_data.get(chunk_id)

    def fake_get_embedding_for_chunk(chunk_id: str) -> list[float] | None:
        call_counts["embedding"] += 1
        return [1.0, 0.0] if chunk_id == "doc-a_chunk_0" else None

    async def fake_search_vector(query_text, top_k, excluded_files, docs_root):
        return [{"chunk_id": chunk_id, "doc_id": "doc-a", "score": 0.7}]

    async def fake_search_keyword(query_text, top_k, excluded_files, docs_root):
        return [{"chunk_id": chunk_id, "doc_id": "doc-a", "score": 0.6}]

    class FakePipeline:
        def process(self, fused, get_embedding, get_content, query, top_n):
            assert get_content(chunk_id) == "Alpha content"
            assert get_content(chunk_id) == "Alpha content"
            assert get_embedding(chunk_id) == [1.0, 0.0]
            assert get_embedding(chunk_id) == [1.0, 0.0]
            return (
                [(chunk_id, fused[0][1])],
                CompressionStats(
                    original_count=len(fused),
                    after_threshold=len(fused),
                    after_content_dedup=len(fused),
                    after_ngram_dedup=len(fused),
                    after_dedup=len(fused),
                    after_doc_limit=len(fused),
                    clusters_merged=0,
                ),
            )

    monkeypatch.setattr(orchestrator, "_search_vector", fake_search_vector)
    monkeypatch.setattr(orchestrator, "_search_keyword", fake_search_keyword)
    monkeypatch.setattr(orchestrator._vector, "get_chunk_by_id", fake_get_chunk_by_id)
    monkeypatch.setattr(
        orchestrator._vector,
        "get_embedding_for_chunk",
        fake_get_embedding_for_chunk,
    )
    monkeypatch.setattr(orchestrator, "_get_pipeline", lambda: FakePipeline())
    monkeypatch.setattr(
        "src.search.orchestrator.expand_query_with_tags",
        lambda **kwargs: [],
    )
    monkeypatch.setattr(
        orchestrator,
        "_get_ranked_graph_neighbors",
        lambda seed_scores: [],
    )
    monkeypatch.setattr(
        orchestrator,
        "_apply_community_boost",
        lambda fused, seed_doc_ids, chunk_id_to_doc_id, result_provenance=None: fused,
    )

    results, _, strategy_stats = await orchestrator.query(
        "alpha",
        top_k=5,
        top_n=1,
        project_filter=["project-a"],
    )

    assert [result.chunk_id for result in results] == [chunk_id]
    assert results[0].content == "Alpha content"
    assert strategy_stats.vector_count == 1
    assert strategy_stats.keyword_count == 1
    assert call_counts == {"chunk": 1, "embedding": 1}
    assert orchestrator._last_query_execution_stats == {
        "metadata_lookups": 1,
        "metadata_cache_hits": 4,
        "content_lookups": 1,
        "content_cache_hits": 1,
        "embedding_fetches": 1,
        "embedding_cache_hits": 1,
        "parent_lookups": 0,
        "parent_cache_hits": 0,
    }


def test_skip_expensive_factual_enrichments_with_single_clear_candidate() -> None:
    orchestrator = _orchestrator()

    assert orchestrator._should_skip_expensive_factual_enrichments(
        classify_query("get_user_by_id"),
        [_result("doc-a_chunk_0", 0.8)],
        [_result("doc-a_chunk_0", 0.7)],
    )


def test_skip_expensive_factual_enrichments_with_small_consensus_set() -> None:
    orchestrator = _orchestrator()

    assert orchestrator._should_skip_expensive_factual_enrichments(
        classify_query("src/mcp_server.py list_tools"),
        [_result("doc-a_chunk_0", 0.8), _result("doc-b_chunk_0", 0.4)],
        [_result("doc-b_chunk_0", 0.9), _result("doc-a_chunk_0", 0.6)],
    )


def test_do_not_skip_expensive_factual_enrichments_for_broad_candidate_set() -> None:
    orchestrator = _orchestrator()

    assert not orchestrator._should_skip_expensive_factual_enrichments(
        classify_query("src/mcp_server.py list_tools"),
        [
            _result("doc-a_chunk_0", 0.9),
            _result("doc-b_chunk_0", 0.8),
            _result("doc-c_chunk_0", 0.7),
            _result("doc-d_chunk_0", 0.6),
        ],
        [
            _result("doc-e_chunk_0", 0.9),
            _result("doc-f_chunk_0", 0.8),
            _result("doc-g_chunk_0", 0.7),
            _result("doc-h_chunk_0", 0.6),
        ],
    )


@pytest.mark.asyncio
async def test_query_skips_tag_expansion_and_reranking_for_clear_factual_query(
    monkeypatch,
) -> None:
    orchestrator = _orchestrator()
    chunk_id = "doc-a_chunk_0"

    async def fake_search_vector(query_text, top_k, excluded_files, docs_root):
        return [_result(chunk_id, 0.8)]

    async def fake_search_keyword(query_text, top_k, excluded_files, docs_root):
        return [_result(chunk_id, 0.9)]

    def fail_if_tag_expansion_called(**kwargs):
        raise AssertionError("tag expansion should be skipped")

    monkeypatch.setattr(orchestrator, "_search_vector", fake_search_vector)
    monkeypatch.setattr(orchestrator, "_search_keyword", fake_search_keyword)
    monkeypatch.setattr(
        "src.search.orchestrator.expand_query_with_tags",
        fail_if_tag_expansion_called,
    )
    monkeypatch.setattr(
        orchestrator,
        "_get_ranked_graph_neighbors",
        lambda seed_scores: [],
    )
    monkeypatch.setattr(
        orchestrator,
        "_apply_community_boost",
        lambda fused, seed_doc_ids, chunk_id_to_doc_id, result_provenance=None: fused,
    )
    monkeypatch.setattr(
        orchestrator,
        "_apply_project_uplift",
        lambda fused, **kwargs: fused,
    )
    monkeypatch.setattr(
        orchestrator,
        "_apply_project_filter",
        lambda fused, **kwargs: fused,
    )
    monkeypatch.setattr(
        orchestrator,
        "_expand_to_parents",
        lambda results, **kwargs: results,
    )
    monkeypatch.setattr(
        orchestrator,
        "_materialize_chunk_results",
        lambda final, **kwargs: [],
    )

    observed_reranking: dict[str, bool] = {"enabled": True}

    class FakePipeline:
        def __init__(self, reranking_enabled: bool) -> None:
            observed_reranking["enabled"] = reranking_enabled

        def process(self, fused, get_embedding, get_content, query, top_n):
            return (
                fused[:top_n],
                CompressionStats(
                    original_count=len(fused),
                    after_threshold=len(fused),
                    after_content_dedup=len(fused),
                    after_ngram_dedup=len(fused),
                    after_dedup=len(fused),
                    after_doc_limit=len(fused),
                    clusters_merged=0,
                ),
            )

    monkeypatch.setattr(
        "src.search.orchestrator.SearchPipeline",
        lambda config: FakePipeline(config.reranking_enabled),
    )

    await orchestrator.query("get_user_by_id", top_k=5, top_n=1)

    assert observed_reranking["enabled"] is False