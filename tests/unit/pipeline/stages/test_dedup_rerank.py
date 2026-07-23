from searchkernel.pipeline.stage import SearchContext
from searchkernel.pipeline.stages.dedup_rerank import DedupRerankStage
from searchkernel.search.pipeline import SearchPipeline, SearchPipelineConfig

_CONTENT = {
    "chunk_a": "Python is a versatile programming language used for web development",
    "chunk_b": "Machine learning algorithms require large datasets for training",
    "chunk_c": "Database indexing improves query performance in production systems",
}


def _get_content(chunk_id: str) -> str:
    return _CONTENT.get(chunk_id, f"content for {chunk_id}")


def _get_embedding(chunk_id: str):
    return [0.1, 0.2, 0.3]


def _fused():
    return [("chunk_a", 0.9), ("chunk_b", 0.7), ("chunk_c", 0.5)]


def _context(top_n: int = 5) -> SearchContext:
    return SearchContext(
        query="query",
        candidates=_fused(),
        metadata={
            "get_embedding": _get_embedding,
            "get_content": _get_content,
            "top_n": top_n,
        },
    )


def test_dedup_rerank_stage_matches_search_pipeline_directly():
    config = SearchPipelineConfig(reranking_enabled=False)

    expected_final, expected_stats = SearchPipeline(config).process(
        _fused(), _get_embedding, _get_content, "query", 5
    )

    result = DedupRerankStage(config).run(_context(top_n=5))

    assert result.candidates == expected_final
    assert result.metadata["compression_stats"] == expected_stats


def test_dedup_rerank_stage_does_not_mutate_input_context():
    config = SearchPipelineConfig(reranking_enabled=False)
    context = _context()

    DedupRerankStage(config).run(context)

    assert context.candidates == _fused()
    assert "compression_stats" not in context.metadata


def test_dedup_rerank_stage_preserves_query():
    config = SearchPipelineConfig(reranking_enabled=False)

    result = DedupRerankStage(config).run(_context())

    assert result.query == "query"
