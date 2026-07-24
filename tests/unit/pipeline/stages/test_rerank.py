from collections.abc import Iterable
from dataclasses import dataclass

from searchkernel.pipeline.stage import SearchContext
from searchkernel.pipeline.stages.rerank import RerankStage
from searchkernel.search.reranker import ReRanker

_CONTENT = {
    "chunk_a": "Short",
    "chunk_b": "A medium length piece of content",
    "chunk_c": "A much longer piece of content than the others here",
}


@dataclass
class _FakeCrossEncoder:
    def predict(
        self,
        sentences: list[tuple[str, str]]
        | list[list[str]]
        | tuple[str, str]
        | list[str],
    ) -> Iterable[float]:
        return [len(content) * 0.01 for _, content in sentences]


def _get_content(chunk_id: str) -> str:
    return _CONTENT[chunk_id]


def _candidates():
    return [("chunk_a", 0.9), ("chunk_b", 0.7), ("chunk_c", 0.5)]


def _context() -> SearchContext:
    return SearchContext(
        query="query",
        candidates=_candidates(),
        metadata={"get_content": _get_content},
    )


def _stage_with_fake(enabled: bool = True, top_n: int = 10) -> RerankStage:
    stage = RerankStage(enabled=enabled, top_n=top_n)
    stage._reranker = ReRanker(model_name="test-model")
    stage._reranker._model = _FakeCrossEncoder()
    return stage


def test_rerank_stage_matches_reranker_directly():
    reranker = ReRanker(model_name="test-model")
    reranker._model = _FakeCrossEncoder()
    expected = reranker.rerank("query", _candidates(), _get_content, 10)

    result = _stage_with_fake().run(_context())

    assert result.candidates == expected


def test_rerank_stage_disabled_is_noop():
    result = RerankStage(enabled=False).run(_context())

    assert result.candidates == _candidates()


def test_rerank_stage_empty_candidates_is_noop():
    context = SearchContext(
        query="q", candidates=[], metadata={"get_content": _get_content}
    )

    result = _stage_with_fake().run(context)

    assert result.candidates == []


def test_rerank_stage_does_not_mutate_input_context():
    context = _context()

    RerankStage(enabled=False).run(context)

    assert context.candidates == _candidates()
