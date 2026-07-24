from searchkernel.pipeline.stage import SearchContext
from searchkernel.pipeline.stages.project_filter import ProjectFilterStage


def _context(candidates, **metadata) -> SearchContext:
    return SearchContext(query="", candidates=candidates, metadata=metadata)


def _chunk_lookup(chunks):
    def get_chunk(chunk_id):
        return chunks.get(chunk_id)

    return get_chunk


def test_project_filter_keeps_only_matching_project():
    chunks = {
        "a_chunk_0": {"metadata": {"project_id": "proj-a"}},
        "b_chunk_0": {"metadata": {"project_id": "proj-b"}},
    }
    context = _context(
        [("a_chunk_0", 0.5), ("b_chunk_0", 0.4)],
        project_filter=["proj-a"],
    )

    result = ProjectFilterStage(_chunk_lookup(chunks)).run(context)

    assert result.candidates == [("a_chunk_0", 0.5)]


def test_project_filter_empty_filter_returns_context_unchanged():
    context = _context([("a_chunk_0", 0.5)], project_filter=[])

    result = ProjectFilterStage(_chunk_lookup({})).run(context)

    assert result.candidates == [("a_chunk_0", 0.5)]


def test_project_filter_no_filter_key_returns_context_unchanged():
    context = _context([("a_chunk_0", 0.5)])

    result = ProjectFilterStage(_chunk_lookup({})).run(context)

    assert result.candidates == [("a_chunk_0", 0.5)]
