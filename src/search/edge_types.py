from enum import Enum


class EdgeType(Enum):
    LINKS_TO = "links_to"
    IMPLEMENTS = "implements"
    TESTS = "tests"
    RELATED = "related"


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
