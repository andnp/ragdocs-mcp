from enum import Enum


class EdgeType(Enum):
    LINKS_TO = "links_to"
    IMPLEMENTS = "implements"
    TESTS = "tests"
    RELATED = "related"


EDGE_TYPE_WEIGHTS: dict[EdgeType, float] = {
    EdgeType.IMPLEMENTS: 1.0,
    EdgeType.LINKS_TO: 0.85,
    EdgeType.RELATED: 0.7,
    EdgeType.TESTS: 0.55,
}


_EDGE_TYPE_ALIASES: dict[str, EdgeType] = {
    "implementation": EdgeType.IMPLEMENTS,
    "implements": EdgeType.IMPLEMENTS,
    "links_to": EdgeType.LINKS_TO,
    "link": EdgeType.LINKS_TO,
    "related": EdgeType.RELATED,
    "related_to": EdgeType.RELATED,
    "test": EdgeType.TESTS,
    "tests": EdgeType.TESTS,
    "transclusion": EdgeType.LINKS_TO,
}


HEADER_TO_EDGE_TYPE: dict[str, EdgeType] = {
    "testing": EdgeType.TESTS,
    "tests": EdgeType.TESTS,
    "test": EdgeType.TESTS,
    "implementation": EdgeType.IMPLEMENTS,
    "implements": EdgeType.IMPLEMENTS,
    "code": EdgeType.IMPLEMENTS,
    "related": EdgeType.RELATED,
    "see also": EdgeType.RELATED,
    "references": EdgeType.RELATED,
}


def normalize_edge_type(edge_type: str | EdgeType):
    if isinstance(edge_type, EdgeType):
        return edge_type

    normalized = edge_type.strip().lower().replace("-", "_").replace(" ", "_")
    return _EDGE_TYPE_ALIASES.get(normalized, EdgeType.LINKS_TO)


def edge_type_weight(edge_type: str | EdgeType):
    return EDGE_TYPE_WEIGHTS[normalize_edge_type(edge_type)]


def infer_edge_type(header_context: str, target: str):
    if not header_context:
        return EdgeType.LINKS_TO

    header_lower = header_context.lower()

    for keyword, edge_type in HEADER_TO_EDGE_TYPE.items():
        if keyword in header_lower:
            return edge_type

    if target.startswith("tests/") or target.startswith("test_"):
        return EdgeType.TESTS

    return EdgeType.LINKS_TO
