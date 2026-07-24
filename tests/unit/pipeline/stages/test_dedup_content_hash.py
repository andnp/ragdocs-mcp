from searchkernel.pipeline.stage import SearchContext
from searchkernel.pipeline.stages.dedup_content_hash import ContentHashDedupStage
from searchkernel.search.dedup import deduplicate_by_content_hash

_CONTENT = {
    "chunk_a": "same content",
    "chunk_b": "same content",
    "chunk_c": "different content",
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


def test_content_hash_dedup_stage_matches_function_directly():
    expected, _removed = deduplicate_by_content_hash(_candidates(), _get_content)

    result = ContentHashDedupStage().run(_context())

    assert result.candidates == expected


def test_content_hash_dedup_stage_does_not_mutate_input_context():
    context = _context()

    ContentHashDedupStage().run(context)

    assert context.candidates == _candidates()
