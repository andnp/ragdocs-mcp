from searchkernel.pipeline.stage import SearchContext
from searchkernel.pipeline.stages.provenance import ProvenanceStage


def _context(strategy_results: dict[str, list[tuple[str, float]]]) -> SearchContext:
    return SearchContext(query="", strategy_results=strategy_results)


def test_provenance_stage_records_rank_and_raw_score_per_strategy():
    context = _context(
        {
            "semantic": [("chunk_a", 0.9), ("chunk_b", 0.5)],
            "keyword": [("chunk_b", 0.7)],
        }
    )

    result = ProvenanceStage().run(context)

    provenance = result.metadata["result_provenance"]
    assert set(provenance) == {"chunk_a", "chunk_b"}
    assert provenance["chunk_a"].strategies == ("semantic",)
    assert provenance["chunk_a"].strategy_details["semantic"].rank == 1
    assert provenance["chunk_a"].strategy_details["semantic"].raw_score == 0.9

    assert provenance["chunk_b"].strategies == ("semantic", "keyword")
    assert provenance["chunk_b"].strategy_details["semantic"].rank == 2
    assert provenance["chunk_b"].strategy_details["keyword"].rank == 1
    assert provenance["chunk_b"].strategy_details["keyword"].raw_score == 0.7


def test_provenance_stage_first_strategy_wins_on_duplicate_chunk_in_same_strategy():
    context = _context({"semantic": [("chunk_a", 0.9), ("chunk_a", 0.1)]})

    result = ProvenanceStage().run(context)

    detail = result.metadata["result_provenance"]["chunk_a"].strategy_details["semantic"]
    assert detail.rank == 1
    assert detail.raw_score == 0.9


def test_provenance_stage_empty_strategy_results_yields_empty_provenance():
    context = _context({})

    result = ProvenanceStage().run(context)

    assert result.metadata["result_provenance"] == {}


def test_provenance_stage_does_not_mutate_input_context():
    context = _context({"semantic": [("chunk_a", 0.9)]})

    ProvenanceStage().run(context)

    assert "result_provenance" not in context.metadata


def test_provenance_stage_prefers_metadata_strategy_results_override():
    context = SearchContext(
        query="",
        strategy_results={"semantic": [("chunk_a", 0.9)]},
        metadata={
            "provenance_strategy_results": {
                "semantic": [("chunk_a", 0.9)],
                "tag_expansion": [("chunk_b", 0.3)],
            }
        },
    )

    result = ProvenanceStage().run(context)

    provenance = result.metadata["result_provenance"]
    assert set(provenance) == {"chunk_a", "chunk_b"}
    assert provenance["chunk_b"].strategies == ("tag_expansion",)
