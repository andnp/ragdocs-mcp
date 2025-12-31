import re
from enum import Enum, auto


class QueryType(Enum):
    FACTUAL = auto()
    NAVIGATIONAL = auto()
    EXPLORATORY = auto()


_CAMEL_CASE_PATTERN = re.compile(r'[a-z][A-Z]|[A-Z]{2,}[a-z]')
_SNAKE_CASE_PATTERN = re.compile(r'\b[a-z]+_[a-z_]+\b')
_BACKTICK_PATTERN = re.compile(r'`[^`]+`')
_VERSION_PATTERN = re.compile(r'\b[vV]?\d+\.\d+(?:\.\d+)?(?:-\w+)?\b')
_QUOTED_PHRASE_PATTERN = re.compile(r'"[^"]+"|\'[^\']+\'')

_NAVIGATIONAL_KEYWORDS = frozenset([
    'section', 'chapter', 'guide', 'tutorial', 'documentation',
    'doc', 'docs', 'page', 'article', 'overview',
])
_NAVIGATIONAL_PHRASES = [
    re.compile(r'\bin\s+the\s+\w+', re.IGNORECASE),
    re.compile(r'\[\[.+?\]\]'),
]

_QUESTION_WORDS = frozenset([
    'what', 'how', 'why', 'when', 'where', 'which', 'who', 'whom',
    'explain', 'describe', 'tell',
])


def _has_factual_signals(query: str) -> bool:
    if _CAMEL_CASE_PATTERN.search(query):
        return True
    if _SNAKE_CASE_PATTERN.search(query):
        return True
    if _BACKTICK_PATTERN.search(query):
        return True
    if _VERSION_PATTERN.search(query):
        return True
    if _QUOTED_PHRASE_PATTERN.search(query):
        return True
    return False


def _has_navigational_signals(query: str) -> bool:
    words = set(query.lower().split())
    if words & _NAVIGATIONAL_KEYWORDS:
        return True
    for pattern in _NAVIGATIONAL_PHRASES:
        if pattern.search(query):
            return True
    return False


def _has_exploratory_signals(query: str) -> bool:
    words = query.lower().split()
    if not words:
        return False
    if words[0] in _QUESTION_WORDS:
        return True
    if query.strip().endswith('?'):
        return True
    return False


def classify_query(query: str) -> QueryType:
    if _has_factual_signals(query):
        return QueryType.FACTUAL
    if _has_navigational_signals(query):
        return QueryType.NAVIGATIONAL
    if _has_exploratory_signals(query):
        return QueryType.EXPLORATORY
    return QueryType.EXPLORATORY


def get_adaptive_weights(
    query_type: QueryType,
    base_semantic: float,
    base_keyword: float,
    base_graph: float,
) -> tuple[float, float, float]:
    if query_type == QueryType.FACTUAL:
        return (base_semantic, base_keyword * 1.5, base_graph)
    if query_type == QueryType.NAVIGATIONAL:
        return (base_semantic, base_keyword, base_graph * 1.5)
    return (base_semantic * 1.3, base_keyword, base_graph)
