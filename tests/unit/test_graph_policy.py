from searchkernel.domain.graph_policy import (
    DEFAULT_GRAPH_PROVIDER,
    EdgeType,
    GraphProvider,
)


def test_default_graph_provider_prioritizes_precise_links():
    weights = DEFAULT_GRAPH_PROVIDER.edge_type_weights
    assert weights[EdgeType.IMPLEMENTS] > weights[EdgeType.LINKS_TO]
    assert weights[EdgeType.LINKS_TO] > weights[EdgeType.RELATED]
    assert weights[EdgeType.RELATED] > weights[EdgeType.TESTS]


def test_weight_for_returns_the_configured_weight():
    assert DEFAULT_GRAPH_PROVIDER.weight_for(EdgeType.IMPLEMENTS) == 1.0


def test_graph_provider_is_pure_config_overridable_without_touching_code():
    custom = GraphProvider(edge_type_weights={EdgeType.TESTS: 5.0})
    assert custom.weight_for(EdgeType.TESTS) == 5.0
