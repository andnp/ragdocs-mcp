from searchkernel.pipeline.stage import SearchContext
from searchkernel.pipeline.stages.routing import RoutingStage
from searchkernel.search.classifier import classify_query, get_adaptive_weights


def _context(query: str) -> SearchContext:
    return SearchContext(
        query=query,
        metadata={
            "base_semantic_weight": 0.6,
            "base_keyword_weight": 0.3,
            "base_graph_weight": 1.0,
        },
    )


def test_routing_stage_matches_classifier_functions_directly():
    query = "what is the getUserById function?"
    expected_type = classify_query(query)
    expected_weights = get_adaptive_weights(expected_type, 0.6, 0.3, 1.0)

    result = RoutingStage().run(_context(query))

    assert result.metadata["query_type"] == expected_type
    assert result.metadata["strategy_weights"] == {
        "semantic": expected_weights[0],
        "keyword": expected_weights[1],
        "graph": expected_weights[2],
    }


def test_routing_stage_does_not_mutate_input_context():
    context = _context("how does this work?")

    RoutingStage().run(context)

    assert "query_type" not in context.metadata
    assert "strategy_weights" not in context.metadata


def test_routing_stage_preserves_candidates():
    context = _context("navigational guide section")
    context.candidates.append(("chunk_a", 0.5))

    result = RoutingStage().run(context)

    assert result.candidates == [("chunk_a", 0.5)]
