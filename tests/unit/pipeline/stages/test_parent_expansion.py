from searchkernel.pipeline.stage import SearchContext
from searchkernel.pipeline.stages.parent_expansion import ParentExpansionStage

from ragdocs.models import SearchResultProvenance


def _context(candidates, **metadata) -> SearchContext:
    return SearchContext(query="", candidates=candidates, metadata=metadata)


def _lookup(chunks):
    def get_chunk(chunk_id):
        return chunks.get(chunk_id)

    return get_chunk


def test_parent_expansion_replaces_child_with_parent():
    chunks = {
        "child": {"metadata": {"parent_chunk_id": "parent"}},
        "parent": {"metadata": {}},
    }
    context = _context([("child", 0.5)])

    result = ParentExpansionStage(_lookup(chunks), _lookup(chunks)).run(context)

    assert result.candidates == [("parent", 0.5)]
    assert result.metadata["missing_chunk_ids"] == []
    assert result.metadata["missing_parent_chunk_ids"] == []


def test_parent_expansion_no_parent_keeps_chunk():
    chunks = {"child": {"metadata": {}}}
    context = _context([("child", 0.5)])

    result = ParentExpansionStage(_lookup(chunks), _lookup(chunks)).run(context)

    assert result.candidates == [("child", 0.5)]


def test_parent_expansion_dedups_shared_parent():
    chunks = {
        "child_a": {"metadata": {"parent_chunk_id": "parent"}},
        "child_b": {"metadata": {"parent_chunk_id": "parent"}},
        "parent": {"metadata": {}},
    }
    context = _context([("child_a", 0.5), ("child_b", 0.4)])

    result = ParentExpansionStage(_lookup(chunks), _lookup(chunks)).run(context)

    assert result.candidates == [("parent", 0.5)]


def test_parent_expansion_reports_missing_chunk():
    context = _context([("missing", 0.5)])

    result = ParentExpansionStage(_lookup({}), _lookup({})).run(context)

    assert result.candidates == [("missing", 0.5)]
    assert result.metadata["missing_chunk_ids"] == ["missing"]


def test_parent_expansion_reports_missing_parent():
    chunks = {"child": {"metadata": {"parent_chunk_id": "parent"}}}
    context = _context([("child", 0.5)])

    result = ParentExpansionStage(_lookup(chunks), _lookup({})).run(context)

    assert result.candidates == [("child", 0.5)]
    assert result.metadata["missing_parent_chunk_ids"] == ["parent"]


def test_parent_expansion_records_provenance_for_expanded_parent():
    chunks = {
        "child": {"metadata": {"parent_chunk_id": "parent"}},
        "parent": {"metadata": {}},
    }
    provenance = SearchResultProvenance()
    provenance.add_strategy("semantic", rank=1, raw_score=0.9)
    result_provenance = {"child": provenance}
    context = _context([("child", 0.5)], result_provenance=result_provenance)

    ParentExpansionStage(_lookup(chunks), _lookup(chunks)).run(context)

    assert result_provenance["parent"].parent_expanded_from == "child"
