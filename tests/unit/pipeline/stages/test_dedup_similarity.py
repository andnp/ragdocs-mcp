from searchkernel.pipeline.stage import SearchContext
from searchkernel.pipeline.stages.dedup_similarity import SimilarityDedupStage
from searchkernel.search.dedup import deduplicate_by_similarity

_EMBEDDINGS = {
    "chunk_a": [1.0, 0.0, 0.0],
    "chunk_b": [1.0, 0.0, 0.0],
    "chunk_c": [0.0, 1.0, 0.0],
}


def _get_embedding(chunk_id: str):
    return _EMBEDDINGS[chunk_id]


def _candidates():
    return [("chunk_a", 0.9), ("chunk_b", 0.8), ("chunk_c", 0.7)]


def _context() -> SearchContext:
    return SearchContext(
        query="q",
        candidates=_candidates(),
        metadata={"get_embedding": _get_embedding},
    )


def test_similarity_dedup_stage_matches_function_directly():
    expected, _clusters_merged = deduplicate_by_similarity(
        _candidates(), _get_embedding, 0.9
    )

    result = SimilarityDedupStage(0.9).run(_context())

    assert result.candidates == expected


def test_similarity_dedup_stage_is_noop_below_two_candidates():
    single = [("chunk_a", 0.9)]
    context = SearchContext(
        query="q", candidates=single, metadata={"get_embedding": _get_embedding}
    )

    result = SimilarityDedupStage(0.9).run(context)

    assert result.candidates == single


def test_similarity_dedup_stage_does_not_mutate_input_context():
    context = _context()

    SimilarityDedupStage(0.9).run(context)

    assert context.candidates == _candidates()
