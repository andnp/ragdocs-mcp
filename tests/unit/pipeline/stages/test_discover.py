from searchkernel.pipeline.stage import SearchContext
from searchkernel.pipeline.stages.discover import DiscoverStage


def _context(**metadata) -> SearchContext:
    return SearchContext(query="", metadata=metadata)


def test_discover_stage_single_root_uses_documents_path(tmp_path):
    (tmp_path / "a.md").write_text("# A")
    (tmp_path / "b.md").write_text("# B")

    result = DiscoverStage().run(
        _context(documents_path=tmp_path, documents_roots=[tmp_path])
    )

    discovered = result.metadata["discovered_files"]
    assert len(discovered) == 2
    assert any("a.md" in f for f in discovered)
    assert any("b.md" in f for f in discovered)


def test_discover_stage_multi_root_unions_all_roots(tmp_path):
    root_a = tmp_path / "a"
    root_b = tmp_path / "b"
    root_a.mkdir()
    root_b.mkdir()
    (root_a / "one.md").write_text("# One")
    (root_b / "two.md").write_text("# Two")

    result = DiscoverStage().run(
        _context(
            documents_path=str(root_a),
            documents_roots=[root_a, root_b],
        )
    )

    discovered = result.metadata["discovered_files"]
    assert any("one.md" in f for f in discovered)
    assert any("two.md" in f for f in discovered)


def test_discover_stage_respects_exclude_patterns(tmp_path):
    (tmp_path / "keep.md").write_text("# Keep")
    (tmp_path / "skip.md").write_text("# Skip")

    result = DiscoverStage().run(
        _context(
            documents_path=tmp_path,
            documents_roots=[tmp_path],
            exclude_patterns=["*skip.md"],
        )
    )

    discovered = result.metadata["discovered_files"]
    assert any("keep.md" in f for f in discovered)
    assert not any("skip.md" in f for f in discovered)


def test_discover_stage_does_not_mutate_input_context(tmp_path):
    context = _context(documents_path=tmp_path, documents_roots=[tmp_path])

    DiscoverStage().run(context)

    assert "discovered_files" not in context.metadata
