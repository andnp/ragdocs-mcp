from searchkernel.pipeline.stage import SearchContext
from searchkernel.pipeline.stages.doc_limit import DocLimitStage
from searchkernel.search.filters import limit_per_document


def _candidates():
    return [
        ("doc_a_chunk_0", 0.9),
        ("doc_a_chunk_1", 0.8),
        ("doc_a_chunk_2", 0.7),
        ("doc_b_chunk_0", 0.6),
    ]


def test_doc_limit_stage_matches_function_directly():
    expected = limit_per_document(_candidates(), 2)

    context = SearchContext(query="q", candidates=_candidates())
    result = DocLimitStage(2).run(context)

    assert result.candidates == expected


def test_doc_limit_stage_zero_keeps_all():
    context = SearchContext(query="q", candidates=_candidates())
    result = DocLimitStage(0).run(context)

    assert result.candidates == _candidates()


def test_doc_limit_stage_does_not_mutate_input_context():
    context = SearchContext(query="q", candidates=_candidates())

    DocLimitStage(2).run(context)

    assert context.candidates == _candidates()
