from pathlib import Path

import pytest

from searchkernel.pipeline.stage import AsyncSearchStage, SearchContext
from searchkernel.pipeline.stages.retrieve import RetrieveStage


def _make_stage(vector_results, keyword_results, calls):
    async def search_vector(query, top_k, excluded_files, docs_root):
        calls.append(("vector", query, top_k, excluded_files, docs_root))
        return vector_results

    async def search_keyword(query, top_k, excluded_files, docs_root):
        calls.append(("keyword", query, top_k, excluded_files, docs_root))
        return keyword_results

    return RetrieveStage(search_vector, search_keyword)


def test_retrieve_stage_is_an_async_search_stage():
    stage = _make_stage([], [], [])

    assert isinstance(stage, AsyncSearchStage)


@pytest.mark.asyncio
async def test_retrieve_stage_writes_both_result_lists_to_metadata():
    vector_results = [{"chunk_id": "v1", "doc_id": "d1", "score": 0.9}]
    keyword_results = [{"chunk_id": "k1", "doc_id": "d2", "score": 0.5}]
    calls: list[tuple] = []
    stage = _make_stage(vector_results, keyword_results, calls)
    docs_root = Path("/docs")

    context = SearchContext(
        query="hello",
        metadata={"top_k": 10, "excluded_files": {"a.md"}, "docs_root": docs_root},
    )

    result = await stage.run(context)

    assert result.metadata["vector_results"] == vector_results
    assert result.metadata["keyword_results"] == keyword_results


@pytest.mark.asyncio
async def test_retrieve_stage_calls_both_searchers_with_shared_params():
    calls: list[tuple] = []
    stage = _make_stage([], [], calls)
    docs_root = Path("/docs")

    context = SearchContext(
        query="hello",
        metadata={"top_k": 7, "excluded_files": None, "docs_root": docs_root},
    )

    await stage.run(context)

    assert ("vector", "hello", 7, None, docs_root) in calls
    assert ("keyword", "hello", 7, None, docs_root) in calls


@pytest.mark.asyncio
async def test_retrieve_stage_does_not_mutate_input_context():
    stage = _make_stage([{"chunk_id": "v1", "doc_id": "d1", "score": 0.9}], [], [])
    context = SearchContext(
        query="hello",
        metadata={"top_k": 10, "docs_root": Path("/docs")},
    )

    await stage.run(context)

    assert context.metadata == {"top_k": 10, "docs_root": Path("/docs")}


@pytest.mark.asyncio
async def test_retrieve_stage_defaults_excluded_files_to_none():
    calls: list[tuple] = []
    stage = _make_stage([], [], calls)

    await stage.run(
        SearchContext(query="q", metadata={"top_k": 5, "docs_root": Path("/docs")})
    )

    assert calls[0][3] is None
    assert calls[1][3] is None
