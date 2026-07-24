from pathlib import Path

import pytest

from searchkernel.pipeline.stage import SearchContext
from searchkernel.pipeline.stages.rag_fusion import RAGFusionConfig, RAGFusionStage


def _context(query: str = "original query") -> SearchContext:
    return SearchContext(
        query=query,
        metadata={
            "top_k": 5,
            "excluded_files": None,
            "docs_root": Path("/docs"),
        },
    )


@pytest.mark.asyncio
async def test_rag_fusion_stage_is_a_noop_passthrough_when_disabled():
    async def generate_query_variants(query: str, num_variants: int) -> list[str]:
        raise AssertionError("must not be called when disabled")

    async def retrieve(query, top_k, excluded_files, docs_root):
        raise AssertionError("must not be called when disabled")

    context = _context()
    result = await RAGFusionStage(
        generate_query_variants, retrieve, RAGFusionConfig(enabled=False)
    ).run(context)

    assert result is context


@pytest.mark.asyncio
async def test_rag_fusion_stage_retrieves_over_original_plus_variants_and_fuses():
    seen_queries: list[str] = []

    async def generate_query_variants(query: str, num_variants: int) -> list[str]:
        assert num_variants == 2
        return [f"{query} variant {i}" for i in range(num_variants)]

    async def retrieve(query, top_k, excluded_files, docs_root):
        seen_queries.append(query)
        if query == "original query":
            return [
                {"chunk_id": "a", "doc_id": "doc_a", "score": 0.9},
                {"chunk_id": "b", "doc_id": "doc_b", "score": 0.5},
            ]
        return [{"chunk_id": "a", "doc_id": "doc_a", "score": 0.3}]

    context = _context()
    result = await RAGFusionStage(
        generate_query_variants,
        retrieve,
        RAGFusionConfig(enabled=True, num_variants=2),
    ).run(context)

    assert seen_queries == [
        "original query",
        "original query variant 0",
        "original query variant 1",
    ]
    chunk_ids = [chunk_id for chunk_id, _ in result.candidates]
    assert "a" in chunk_ids
    assert "b" in chunk_ids
    # "a" appears in every variant's results, "b" only in the original's --
    # RRF fusion must rank the more consistently-retrieved chunk first.
    assert chunk_ids.index("a") < chunk_ids.index("b")


@pytest.mark.asyncio
async def test_rag_fusion_stage_top_k_and_docs_root_thread_through_to_retrieve():
    calls: list[tuple] = []

    async def generate_query_variants(query: str, num_variants: int) -> list[str]:
        return []

    async def retrieve(query, top_k, excluded_files, docs_root):
        calls.append((query, top_k, excluded_files, docs_root))
        return []

    context = SearchContext(
        query="q",
        metadata={
            "top_k": 7,
            "excluded_files": {"skip.md"},
            "docs_root": Path("/tmp/docs"),
        },
    )
    await RAGFusionStage(
        generate_query_variants, retrieve, RAGFusionConfig(enabled=True)
    ).run(context)

    assert calls == [("q", 7, {"skip.md"}, Path("/tmp/docs"))]
