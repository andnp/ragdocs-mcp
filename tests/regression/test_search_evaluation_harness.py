import pytest

from tests.search.evaluation_harness import (
    FIXTURE_CORPUS_ROOT,
    SEARCH_EVALUATION_CASES,
    build_search_evaluation_harness,
)


@pytest.fixture(scope="module")
def search_evaluation_harness(shared_embedding_model, tmp_path_factory):
    return build_search_evaluation_harness(
        tmp_path_factory.mktemp("search-eval"),
        shared_embedding_model,
    )


def test_search_eval_cases_cover_expected_query_shapes() -> None:
    case_ids = {case.case_id for case in SEARCH_EVALUATION_CASES}

    assert case_ids == {
        "exact_heading",
        "conceptual_refresh_tokens",
        "graph_adjacent_api_auth",
        "artifact_fileish",
        "scoped_project_context",
    }
    assert (FIXTURE_CORPUS_ROOT / "alpha" / "docs").exists()
    assert (FIXTURE_CORPUS_ROOT / "beta" / "docs").exists()


@pytest.mark.asyncio
async def test_search_evaluation_harness_fixture_corpus(search_evaluation_harness) -> None:
    report = await search_evaluation_harness.evaluate()
    failures = report.expectation_failures()

    assert failures == [], report.format_summary()

    aggregate = report.aggregate
    assert aggregate.query_count == 5
    assert aggregate.mrr >= 0.8, report.format_summary()
    assert aggregate.recall_at_3 >= 0.8, report.format_summary()
    assert aggregate.recall_at_5 >= 0.9, report.format_summary()
    assert aggregate.ndcg_at_5 >= 0.85, report.format_summary()

    scoped_case = next(
        case_result
        for case_result in report.case_results
        if case_result.case.case_id == "scoped_project_context"
    )
    assert scoped_case.top_path() == "alpha/docs/project-rollout-checklist.md"
