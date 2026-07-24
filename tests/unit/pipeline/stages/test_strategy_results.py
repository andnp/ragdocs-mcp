from searchkernel.pipeline.stage import SearchContext, SearchStage
from searchkernel.pipeline.stages.strategy_results import StrategyResultsStage


def _context(**metadata) -> SearchContext:
    return SearchContext(query="", metadata=metadata)


def test_strategy_results_stage_is_a_search_stage():
    assert isinstance(StrategyResultsStage(), SearchStage)


def test_builds_narrow_strategy_results_for_fusion():
    context = _context(
        vector_results=[{"chunk_id": "a_chunk_0", "doc_id": "a", "score": 0.9}],
        keyword_results=[{"chunk_id": "b_chunk_0", "doc_id": "b", "score": 0.5}],
        graph_chunk_ids=["c_chunk_0"],
        graph_doc_scores={"c": 0.4},
    )

    result = StrategyResultsStage().run(context)

    assert result.strategy_results == {
        "semantic": [("a_chunk_0", 0.9)],
        "keyword": [("b_chunk_0", 0.5)],
        "graph": [("c_chunk_0", 0.4)],
    }
    assert "tag_expansion" not in result.metadata["provenance_strategy_results"]


def test_provenance_strategy_results_includes_tag_expansion_when_applied():
    context = _context(
        vector_results=[{"chunk_id": "a_chunk_0", "doc_id": "a", "score": 0.9}],
        keyword_results=[],
        graph_chunk_ids=[],
        graph_doc_scores={},
        applied_tag_expansion_results=[
            {"chunk_id": "new_chunk_0", "doc_id": "new", "score": 0.3}
        ],
    )

    result = StrategyResultsStage().run(context)

    assert result.metadata["provenance_strategy_results"]["tag_expansion"] == [
        ("new_chunk_0", 0.3)
    ]
    assert "tag_expansion" not in result.strategy_results


def test_does_not_mutate_input_context():
    context = _context(
        vector_results=[{"chunk_id": "a_chunk_0", "doc_id": "a", "score": 0.9}],
        keyword_results=[],
        graph_chunk_ids=[],
        graph_doc_scores={},
    )

    StrategyResultsStage().run(context)

    assert context.strategy_results == {}
    assert "provenance_strategy_results" not in context.metadata
