from searchkernel.pipeline.stage import SearchContext
from searchkernel.pipeline.stages.dedup_ngram import NgramDedupStage
from searchkernel.search.dedup import deduplicate_by_ngram

_CONTENT = {
    "chunk_a": "Python is a versatile programming language used for web development",
    "chunk_b": "Python is a versatile programming language used for web dev work",
    "chunk_c": "Database indexing improves query performance in production systems",
}


def _get_content(chunk_id: str) -> str:
    return _CONTENT[chunk_id]


def _candidates():
    return [("chunk_a", 0.9), ("chunk_b", 0.8), ("chunk_c", 0.7)]


def _context() -> SearchContext:
    return SearchContext(
        query="q",
        candidates=_candidates(),
        metadata={"get_content": _get_content},
    )


def test_ngram_dedup_stage_matches_function_directly():
    expected, _removed = deduplicate_by_ngram(_candidates(), _get_content, 0.7)

    result = NgramDedupStage(0.7).run(_context())

    assert result.candidates == expected


def test_ngram_dedup_stage_is_noop_below_two_candidates():
    single = [("chunk_a", 0.9)]
    context = SearchContext(
        query="q", candidates=single, metadata={"get_content": _get_content}
    )

    result = NgramDedupStage(0.7).run(context)

    assert result.candidates == single


def test_ngram_dedup_stage_does_not_mutate_input_context():
    context = _context()

    NgramDedupStage(0.7).run(context)

    assert context.candidates == _candidates()
