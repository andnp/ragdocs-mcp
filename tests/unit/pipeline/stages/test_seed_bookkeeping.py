from searchkernel.pipeline.stage import SearchContext, SearchStage
from searchkernel.pipeline.stages.seed_bookkeeping import SeedBookkeepingStage
from searchkernel.search.classifier import QueryType


def _context(**metadata) -> SearchContext:
    return SearchContext(query="", metadata=metadata)


def test_seed_bookkeeping_stage_is_a_search_stage():
    assert isinstance(SeedBookkeepingStage(), SearchStage)


def test_seed_bookkeeping_builds_chunk_and_doc_id_maps():
    context = _context(
        vector_results=[{"chunk_id": "a_chunk_0", "doc_id": "a", "score": 0.9}],
        keyword_results=[{"chunk_id": "b_chunk_0", "doc_id": "b", "score": 0.5}],
        query_type=QueryType.EXPLORATORY,
    )

    result = SeedBookkeepingStage().run(context)

    assert result.metadata["chunk_id_to_doc_id"] == {
        "a_chunk_0": "a",
        "b_chunk_0": "b",
    }
    assert result.metadata["all_doc_ids"] == {"a", "b"}


def test_seed_bookkeeping_seed_scores_takes_best_score_per_doc():
    context = _context(
        vector_results=[{"chunk_id": "a_chunk_0", "doc_id": "a", "score": 0.4}],
        keyword_results=[{"chunk_id": "a_chunk_1", "doc_id": "a", "score": 0.9}],
        query_type=QueryType.EXPLORATORY,
    )

    result = SeedBookkeepingStage().run(context)

    assert result.metadata["seed_scores"] == {"a": 0.9}


def test_seed_bookkeeping_sets_skip_tag_expansion_for_clear_factual_query():
    context = _context(
        vector_results=[{"chunk_id": "a_chunk_0", "doc_id": "a", "score": 0.9}],
        keyword_results=[{"chunk_id": "a_chunk_0", "doc_id": "a", "score": 0.8}],
        query_type=QueryType.FACTUAL,
    )

    result = SeedBookkeepingStage().run(context)

    assert result.metadata["skip_tag_expansion"] is True


def test_seed_bookkeeping_does_not_skip_for_non_factual_query():
    context = _context(
        vector_results=[{"chunk_id": "a_chunk_0", "doc_id": "a", "score": 0.9}],
        keyword_results=[],
        query_type=QueryType.EXPLORATORY,
    )

    result = SeedBookkeepingStage().run(context)

    assert result.metadata["skip_tag_expansion"] is False


def test_seed_bookkeeping_does_not_mutate_input_context():
    context = _context(
        vector_results=[{"chunk_id": "a_chunk_0", "doc_id": "a", "score": 0.9}],
        keyword_results=[],
        query_type=QueryType.EXPLORATORY,
    )

    SeedBookkeepingStage().run(context)

    assert "seed_scores" not in context.metadata
