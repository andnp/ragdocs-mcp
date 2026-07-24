from searchkernel.pipeline.stage import SearchContext, SearchStage
from searchkernel.pipeline.stages.graph_expand import GraphExpandStage


def _make_stage(ranked_neighbors, chunk_ids, calls):
    def rank_neighbors(seed_scores):
        calls.append(("rank_neighbors", seed_scores))
        return ranked_neighbors

    def build_chunk_candidates(neighbor_doc_ids, top_k, excluded_chunk_ids):
        calls.append(("build_chunk_candidates", neighbor_doc_ids, top_k, excluded_chunk_ids))
        return chunk_ids

    return GraphExpandStage(rank_neighbors, build_chunk_candidates)


def test_graph_expand_stage_is_a_search_stage():
    stage = _make_stage([], [], [])

    assert isinstance(stage, SearchStage)


def test_graph_expand_stage_writes_chunk_ids_and_doc_scores_to_metadata():
    ranked_neighbors = [("doc-a", 0.9), ("doc-b", 0.4)]
    chunk_ids = ["doc-a_chunk_0", "doc-b_chunk_0"]
    stage = _make_stage(ranked_neighbors, chunk_ids, [])

    context = SearchContext(
        query="",
        metadata={
            "seed_scores": {"doc-a": 0.9},
            "top_k": 5,
            "excluded_chunk_ids": {"x_chunk_0"},
        },
    )

    result = stage.run(context)

    assert result.metadata["graph_chunk_ids"] == chunk_ids
    assert result.metadata["graph_doc_scores"] == {"doc-a": 0.9, "doc-b": 0.4}


def test_graph_expand_stage_passes_seed_scores_and_top_k_through():
    calls: list[tuple] = []
    stage = _make_stage([("doc-a", 0.9)], [], calls)

    stage.run(
        SearchContext(
            query="",
            metadata={
                "seed_scores": {"doc-a": 0.9},
                "top_k": 7,
                "excluded_chunk_ids": None,
            },
        )
    )

    assert calls[0] == ("rank_neighbors", {"doc-a": 0.9})
    assert calls[1] == ("build_chunk_candidates", ["doc-a"], 7, None)


def test_graph_expand_stage_defaults_excluded_chunk_ids_to_none():
    calls: list[tuple] = []
    stage = _make_stage([], [], calls)

    stage.run(SearchContext(query="", metadata={"seed_scores": {}, "top_k": 5}))

    assert calls[1][3] is None


def test_graph_expand_stage_does_not_mutate_input_context():
    stage = _make_stage([("doc-a", 0.9)], ["doc-a_chunk_0"], [])
    context = SearchContext(
        query="",
        metadata={"seed_scores": {"doc-a": 0.9}, "top_k": 5},
    )

    stage.run(context)

    assert context.metadata == {"seed_scores": {"doc-a": 0.9}, "top_k": 5}
