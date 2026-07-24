"""GraphProvider: domain policy for graph edge-type weighting.

Pure data/config -- no I/O, no adapters -- so the weight a graph edge
type contributes to one-hop expansion scoring is a policy object rather
than a module-level dict hardcoded next to the string-normalization
logic that used to own it (`searchkernel.search.edge_types`, which now
delegates here for its weights).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class EdgeType(Enum):
    LINKS_TO = "links_to"
    IMPLEMENTS = "implements"
    TESTS = "tests"
    RELATED = "related"


@dataclass(frozen=True)
class GraphProvider:
    """Edge-type weight policy consumed by graph one-hop expansion scoring."""

    edge_type_weights: dict[EdgeType, float] = field(
        default_factory=lambda: {
            EdgeType.IMPLEMENTS: 1.0,
            EdgeType.LINKS_TO: 0.85,
            EdgeType.RELATED: 0.7,
            EdgeType.TESTS: 0.55,
        }
    )

    def weight_for(self, edge_type: EdgeType) -> float:
        return self.edge_type_weights[edge_type]


DEFAULT_GRAPH_PROVIDER = GraphProvider()
