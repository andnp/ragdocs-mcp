from searchkernel.pipeline.stage import SearchContext
from searchkernel.pipeline.stages.project_uplift import ProjectUpliftStage


def _context(candidates, **metadata) -> SearchContext:
    return SearchContext(query="", candidates=candidates, metadata=metadata)


def _chunk_lookup(chunks):
    def get_chunk(chunk_id):
        return chunks.get(chunk_id)

    return get_chunk


def test_project_uplift_boosts_matching_project_and_resorts():
    chunks = {
        "a_chunk_0": {"metadata": {"project_id": "proj-a"}},
        "b_chunk_0": {"metadata": {"project_id": "proj-b"}},
    }
    context = _context(
        [("a_chunk_0", 0.4), ("b_chunk_0", 0.4)],
        active_project="proj-a",
    )

    result = ProjectUpliftStage(_chunk_lookup(chunks), 1.2).run(context)

    assert result.candidates[0] == ("a_chunk_0", 0.48)
    assert result.candidates[1] == ("b_chunk_0", 0.4)


def test_project_uplift_no_active_project_returns_context_unchanged():
    context = _context([("a_chunk_0", 0.4)], active_project=None)

    result = ProjectUpliftStage(_chunk_lookup({}), 1.2).run(context)

    assert result.candidates == [("a_chunk_0", 0.4)]


def test_project_uplift_uses_configured_multiplier():
    chunks = {"a_chunk_0": {"metadata": {"project_id": "proj-a"}}}
    context = _context([("a_chunk_0", 0.05)], active_project="proj-a")

    result = ProjectUpliftStage(_chunk_lookup(chunks), 1.5).run(context)

    assert result.candidates == [("a_chunk_0", 0.05 * 1.5)]


def test_project_uplift_records_provenance():
    chunks = {"a_chunk_0": {"metadata": {"project_id": "proj-a"}}}
    provenance = {}
    context = _context(
        [("a_chunk_0", 0.4)],
        active_project="proj-a",
        result_provenance=provenance,
    )

    ProjectUpliftStage(_chunk_lookup(chunks), 1.2).run(context)

    assert provenance["a_chunk_0"].project_uplift == 1.2
