from searchkernel.pipeline.stage import SearchContext
from searchkernel.pipeline.stages.source_filter import SourceFilterStage


def _context(candidates, **metadata) -> SearchContext:
    return SearchContext(query="", candidates=candidates, metadata=metadata)


def _chunk_lookup(chunks):
    def get_chunk(chunk_id):
        return chunks.get(chunk_id)

    return get_chunk


def test_source_filter_keeps_only_matching_source_kind():
    chunks = {
        "a_chunk_0": {"metadata": {"source_kind": "document"}},
        "b_chunk_0": {"metadata": {"source_kind": "git_commit"}},
    }
    context = _context(
        [("a_chunk_0", 0.5), ("b_chunk_0", 0.4)],
        source_filter=["git_commit"],
    )

    result = SourceFilterStage(_chunk_lookup(chunks)).run(context)

    assert result.candidates == [("b_chunk_0", 0.4)]


def test_source_filter_empty_filter_returns_context_unchanged():
    context = _context([("a_chunk_0", 0.5)], source_filter=None)

    result = SourceFilterStage(_chunk_lookup({})).run(context)

    assert result.candidates == [("a_chunk_0", 0.5)]
