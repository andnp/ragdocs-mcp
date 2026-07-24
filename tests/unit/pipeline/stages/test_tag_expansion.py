from searchkernel.pipeline.stage import SearchContext
from searchkernel.pipeline.stages.tag_expansion import TagExpansionStage


def _context(**metadata) -> SearchContext:
    return SearchContext(query="", metadata=metadata)


def test_tag_expansion_merges_new_chunks_into_bookkeeping():
    def fake_expand(initial_results, top_k):
        assert top_k == 5
        return [{"chunk_id": "new_chunk_0", "doc_id": "new", "score": 0.3}]

    context = _context(
        vector_results=[{"chunk_id": "a_chunk_0", "doc_id": "a", "score": 0.9}],
        keyword_results=[],
        chunk_id_to_doc_id={"a_chunk_0": "a"},
        all_doc_ids={"a"},
        top_k=5,
    )

    result = TagExpansionStage(fake_expand).run(context)

    assert result.metadata["tag_expansion_count"] == 1
    assert result.metadata["chunk_id_to_doc_id"] == {"a_chunk_0": "a", "new_chunk_0": "new"}
    assert result.metadata["all_doc_ids"] == {"a", "new"}
    assert [r["chunk_id"] for r in result.metadata["vector_results"]] == [
        "a_chunk_0",
        "new_chunk_0",
    ]
    assert [r["chunk_id"] for r in result.metadata["applied_tag_expansion_results"]] == [
        "new_chunk_0"
    ]


def test_tag_expansion_skips_chunks_already_present():
    def fake_expand(initial_results, top_k):
        return [{"chunk_id": "a_chunk_0", "doc_id": "a", "score": 0.5}]

    context = _context(
        vector_results=[{"chunk_id": "a_chunk_0", "doc_id": "a", "score": 0.9}],
        keyword_results=[],
        chunk_id_to_doc_id={"a_chunk_0": "a"},
        all_doc_ids={"a"},
        top_k=5,
    )

    result = TagExpansionStage(fake_expand).run(context)

    assert result.metadata["tag_expansion_count"] == 0
    assert result.metadata["applied_tag_expansion_results"] == []
    assert len(result.metadata["vector_results"]) == 1


def test_tag_expansion_skip_flag_short_circuits():
    def fail_if_called(initial_results, top_k):
        raise AssertionError("should not be called when skipped")

    context = _context(
        vector_results=[],
        keyword_results=[],
        chunk_id_to_doc_id={},
        all_doc_ids=set(),
        top_k=5,
        skip_tag_expansion=True,
    )

    result = TagExpansionStage(fail_if_called).run(context)

    assert result.metadata["tag_expansion_count"] == 0
    assert result.metadata["vector_results"] == []


def test_tag_expansion_republishes_seed_doc_ids_and_excluded_chunk_ids():
    def fake_expand(initial_results, top_k):
        return [{"chunk_id": "new_chunk_0", "doc_id": "new", "score": 0.3}]

    context = _context(
        vector_results=[{"chunk_id": "a_chunk_0", "doc_id": "a", "score": 0.9}],
        keyword_results=[],
        chunk_id_to_doc_id={"a_chunk_0": "a"},
        all_doc_ids={"a"},
        top_k=5,
    )

    result = TagExpansionStage(fake_expand).run(context)

    assert result.metadata["seed_doc_ids"] == {"a", "new"}
    assert result.metadata["excluded_chunk_ids"] == {"a_chunk_0", "new_chunk_0"}


def test_tag_expansion_does_not_mutate_input_context():
    def fake_expand(initial_results, top_k):
        return [{"chunk_id": "new_chunk_0", "doc_id": "new", "score": 0.3}]

    original_vector_results = [{"chunk_id": "a_chunk_0", "doc_id": "a", "score": 0.9}]
    context = _context(
        vector_results=original_vector_results,
        keyword_results=[],
        chunk_id_to_doc_id={"a_chunk_0": "a"},
        all_doc_ids={"a"},
        top_k=5,
    )

    TagExpansionStage(fake_expand).run(context)

    assert len(original_vector_results) == 1
    assert context.metadata["chunk_id_to_doc_id"] == {"a_chunk_0": "a"}
