from tests.search.evaluation_harness import compute_ranking_metrics


def test_compute_ranking_metrics_uses_document_level_ranks() -> None:
    ranked_doc_ids = ("doc-a", "doc-b", "doc-c")
    relevant_doc_ids = ("doc-b", "doc-c")

    metrics = compute_ranking_metrics(ranked_doc_ids, relevant_doc_ids)

    assert metrics.reciprocal_rank == 0.5
    assert metrics.recall_at_1 == 0.0
    assert metrics.recall_at_3 == 1.0
    assert metrics.recall_at_5 == 1.0
    assert round(metrics.ndcg_at_3, 3) == 0.693
    assert round(metrics.ndcg_at_5, 3) == 0.693
