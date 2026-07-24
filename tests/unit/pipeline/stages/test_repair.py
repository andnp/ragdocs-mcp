from searchkernel.pipeline.stage import SearchContext
from searchkernel.pipeline.stages.repair import RepairStage


def _context(**metadata) -> SearchContext:
    return SearchContext(query="", metadata=metadata)


def test_repair_stage_resolves_via_documents_roots(tmp_path):
    (tmp_path / "guide.md").write_text("# Guide")

    result = RepairStage().run(
        _context(
            doc_id="guide",
            docs_path=tmp_path,
            documents_roots=[tmp_path],
            suffixes=[".md", ".txt"],
        )
    )

    assert result.metadata["resolved_path"] == (tmp_path / "guide.md").resolve()


def test_repair_stage_falls_back_to_docs_path_when_roots_miss(tmp_path):
    other_root = tmp_path / "other"
    other_root.mkdir()
    docs_path = tmp_path / "docs"
    docs_path.mkdir()
    (docs_path / "guide.md").write_text("# Guide")

    result = RepairStage().run(
        _context(
            doc_id="guide",
            docs_path=docs_path,
            documents_roots=[other_root],
            suffixes=[".md", ".txt"],
        )
    )

    assert result.metadata["resolved_path"] == (docs_path / "guide.md").resolve()


def test_repair_stage_returns_none_when_file_missing(tmp_path):
    result = RepairStage().run(
        _context(
            doc_id="missing",
            docs_path=tmp_path,
            documents_roots=[tmp_path],
            suffixes=[".md", ".txt"],
        )
    )

    assert result.metadata["resolved_path"] is None


def test_repair_stage_does_not_mutate_input_context(tmp_path):
    context = _context(
        doc_id="missing",
        docs_path=tmp_path,
        documents_roots=[tmp_path],
        suffixes=[".md"],
    )

    RepairStage().run(context)

    assert "resolved_path" not in context.metadata
