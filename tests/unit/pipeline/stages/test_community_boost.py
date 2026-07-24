from searchkernel.pipeline.stage import SearchContext
from searchkernel.pipeline.stages.community_boost import CommunityBoostStage


def _context(candidates, **metadata) -> SearchContext:
    return SearchContext(query="", candidates=candidates, metadata=metadata)


def test_community_boost_applies_boost_and_resorts():
    def boost_by_community(chunk_doc_ids, seed_doc_ids, factor):
        assert factor == 1.1
        return {"doc-b": 1.1}

    context = _context(
        [("doc-a_chunk_0", 0.4), ("doc-b_chunk_0", 0.4)],
        seed_doc_ids={"doc-a"},
        chunk_id_to_doc_id={"doc-a_chunk_0": "doc-a", "doc-b_chunk_0": "doc-b"},
    )

    result = CommunityBoostStage(boost_by_community).run(context)

    assert result.candidates[0][0] == "doc-b_chunk_0"
    assert round(result.candidates[0][1], 10) == 0.44


def test_community_boost_clamps_score_to_one():
    def boost_by_community(chunk_doc_ids, seed_doc_ids, factor):
        return {"doc-a": 1.1}

    context = _context(
        [("doc-a_chunk_0", 0.95)],
        seed_doc_ids=set(),
        chunk_id_to_doc_id={"doc-a_chunk_0": "doc-a"},
    )

    result = CommunityBoostStage(boost_by_community).run(context)

    assert result.candidates[0][1] == 1.0


def test_community_boost_records_provenance_when_boosted():
    def boost_by_community(chunk_doc_ids, seed_doc_ids, factor):
        return {"doc-a": 1.1}

    provenance = {}
    context = _context(
        [("doc-a_chunk_0", 0.5)],
        seed_doc_ids=set(),
        chunk_id_to_doc_id={"doc-a_chunk_0": "doc-a"},
        result_provenance=provenance,
    )

    CommunityBoostStage(boost_by_community).run(context)

    assert provenance["doc-a_chunk_0"].community_boost == 1.1


def test_community_boost_falls_back_to_chunk_id_split_when_map_missing_entry():
    captured = {}

    def boost_by_community(chunk_doc_ids, seed_doc_ids, factor):
        captured["chunk_doc_ids"] = chunk_doc_ids
        return {}

    context = _context(
        [("doc-a_chunk_0", 0.5)],
        seed_doc_ids=set(),
        chunk_id_to_doc_id={},
    )

    CommunityBoostStage(boost_by_community).run(context)

    assert captured["chunk_doc_ids"] == ["doc-a"]
