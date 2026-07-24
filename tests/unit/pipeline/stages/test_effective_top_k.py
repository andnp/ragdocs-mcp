from searchkernel.pipeline.stage import SearchContext, SearchStage
from searchkernel.pipeline.stages.effective_top_k import EffectiveTopKStage
from searchkernel.search.classifier import QueryType


def _context(**metadata) -> SearchContext:
    return SearchContext(query="", metadata=metadata)


def test_effective_top_k_stage_is_a_search_stage():
    assert isinstance(EffectiveTopKStage(), SearchStage)


def test_contracts_top_k_for_small_factual_queries():
    context = _context(
        requested_top_k=20,
        top_n=3,
        project_filter=None,
        query_type=QueryType.FACTUAL,
    )

    result = EffectiveTopKStage().run(context)

    assert result.metadata["top_k"] == 8


def test_does_not_contract_for_exploratory_queries():
    context = _context(
        requested_top_k=20,
        top_n=3,
        project_filter=None,
        query_type=QueryType.EXPLORATORY,
    )

    result = EffectiveTopKStage().run(context)

    assert result.metadata["top_k"] == 20


def test_does_not_contract_when_top_n_is_large():
    context = _context(
        requested_top_k=20,
        top_n=6,
        project_filter=None,
        query_type=QueryType.FACTUAL,
    )

    result = EffectiveTopKStage().run(context)

    assert result.metadata["top_k"] == 20


def test_does_not_contract_with_a_project_filter():
    context = _context(
        requested_top_k=20,
        top_n=3,
        project_filter=["docs-project"],
        query_type=QueryType.FACTUAL,
    )

    result = EffectiveTopKStage().run(context)

    assert result.metadata["top_k"] == 20


def test_does_not_contract_a_non_positive_top_k():
    context = _context(
        requested_top_k=0,
        top_n=3,
        project_filter=None,
        query_type=QueryType.FACTUAL,
    )

    result = EffectiveTopKStage().run(context)

    assert result.metadata["top_k"] == 0


def test_does_not_mutate_input_context():
    context = _context(
        requested_top_k=20,
        top_n=3,
        project_filter=None,
        query_type=QueryType.FACTUAL,
    )

    EffectiveTopKStage().run(context)

    assert "top_k" not in context.metadata
