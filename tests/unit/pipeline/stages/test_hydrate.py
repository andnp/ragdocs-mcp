from searchkernel.domain import ChunkResult, SearchResultProvenance
from searchkernel.pipeline.stage import SearchContext
from searchkernel.pipeline.stages.hydrate import HydrateStage


def _context(
    candidates: list[tuple[str, float]],
    result_provenance: dict[str, SearchResultProvenance] | None = None,
) -> SearchContext:
    metadata = {}
    if result_provenance is not None:
        metadata["result_provenance"] = result_provenance
    return SearchContext(query="", candidates=candidates, metadata=metadata)


def test_hydrate_stage_wraps_successful_hydration_and_attaches_provenance():
    def hydrate(chunk_id: str, score: float) -> ChunkResult | None:
        return ChunkResult(
            chunk_id=chunk_id,
            record_id="doc1",
            score=score,
            content="body",
            metadata={"header_path": "H", "file_path": "doc1.md"},
        )

    provenance = SearchResultProvenance()
    provenance.add_strategy("semantic", 1, 0.9)
    result = HydrateStage(hydrate).run(
        _context([("doc1_chunk_0", 0.9)], {"doc1_chunk_0": provenance})
    )

    chunk_results = result.metadata["chunk_results"]
    assert len(chunk_results) == 1
    assert chunk_results[0].chunk_id == "doc1_chunk_0"
    assert chunk_results[0].content == "body"
    assert chunk_results[0].provenance is provenance
    assert result.metadata["missing_chunk_ids"] == []


def test_hydrate_stage_emits_placeholder_and_reports_missing_chunk_ids():
    def hydrate(chunk_id: str, score: float) -> ChunkResult | None:
        return None

    result = HydrateStage(hydrate).run(_context([("doc1_chunk_0", 0.5)]))

    chunk_results = result.metadata["chunk_results"]
    assert len(chunk_results) == 1
    placeholder = chunk_results[0]
    assert placeholder.chunk_id == "doc1_chunk_0"
    assert placeholder.record_id == "doc1"
    assert placeholder.content == ""
    assert placeholder.metadata["header_path"] == ""
    assert placeholder.metadata["file_path"] == ""
    assert result.metadata["missing_chunk_ids"] == ["doc1_chunk_0"]


def test_hydrate_stage_preserves_candidate_order_with_mixed_hits_and_misses():
    def hydrate(chunk_id: str, score: float) -> ChunkResult | None:
        if chunk_id == "missing":
            return None
        return ChunkResult(
            chunk_id=chunk_id,
            record_id="doc1",
            score=score,
            content="",
            metadata={"header_path": "", "file_path": ""},
        )

    result = HydrateStage(hydrate).run(
        _context([("a", 0.9), ("missing", 0.5), ("b", 0.1)])
    )

    chunk_ids = [c.chunk_id for c in result.metadata["chunk_results"]]
    assert chunk_ids == ["a", "missing", "b"]
    assert result.metadata["missing_chunk_ids"] == ["missing"]


def test_hydrate_stage_does_not_mutate_input_context():
    context = _context([("doc1_chunk_0", 0.9)])

    HydrateStage(lambda chunk_id, score: None).run(context)

    assert "chunk_results" not in context.metadata
    assert "missing_chunk_ids" not in context.metadata
