from pathlib import Path

from src.config import ChunkingConfig, Config, IndexingConfig, SearchConfig
from src.indices.graph import GraphStore
from src.indices.keyword import KeywordIndex
from src.indices.vector import VectorIndex
from src.search.orchestrator import SearchOrchestrator


def _orchestrator(*, detected_project: str | None) -> SearchOrchestrator:
    config = Config(
        indexing=IndexingConfig(
            documents_path="/tmp/docs",
            index_path="/tmp/index",
        ),
        search=SearchConfig(),
        chunking=ChunkingConfig(),
        detected_project=detected_project,
    )
    return SearchOrchestrator(
        VectorIndex(embedding_model_name="BAAI/bge-small-en-v1.5"),
        KeywordIndex(),
        GraphStore(),
        config,
        None,
        documents_path=Path(config.indexing.documents_path),
    )


def test_project_uplift_boosts_matching_results(monkeypatch) -> None:
    orchestrator = _orchestrator(detected_project="project-a")

    chunk_data = {
        "chunk-a": {"metadata": {"project_id": "project-a"}},
        "chunk-b": {"metadata": {"project_id": "project-b"}},
    }
    monkeypatch.setattr(
        orchestrator._vector,
        "get_chunk_by_id",
        lambda chunk_id: chunk_data.get(chunk_id),
    )

    boosted = orchestrator._apply_project_uplift(
        [("chunk-b", 0.05), ("chunk-a", 0.049)]
    )

    assert boosted[0][0] == "chunk-a"
    assert boosted[0][1] == 0.049 * 1.2
    assert boosted[1] == ("chunk-b", 0.05)


def test_project_uplift_is_noop_without_active_project(monkeypatch) -> None:
    orchestrator = _orchestrator(detected_project=None)
    monkeypatch.setattr(
        orchestrator._vector,
        "get_chunk_by_id",
        lambda chunk_id: {"metadata": {"project_id": "project-a"}},
    )

    fused = [("chunk-a", 0.05)]

    assert orchestrator._apply_project_uplift(fused) == fused


def test_project_uplift_prefers_explicit_project_context(monkeypatch) -> None:
    orchestrator = _orchestrator(detected_project="project-a")
    monkeypatch.setattr(
        orchestrator._vector,
        "get_chunk_by_id",
        lambda chunk_id: {"metadata": {"project_id": "project-b"}},
    )

    boosted = orchestrator._apply_project_uplift(
        [("chunk-b", 0.05)], project_context="project-b"
    )

    assert boosted == [("chunk-b", 0.05 * 1.2)]


def test_project_filter_restricts_results(monkeypatch) -> None:
    orchestrator = _orchestrator(detected_project=None)
    chunk_data = {
        "chunk-a": {"metadata": {"project_id": "project-a"}},
        "chunk-b": {"metadata": {"project_id": "project-b"}},
        "chunk-none": {"metadata": {}},
    }
    monkeypatch.setattr(
        orchestrator._vector,
        "get_chunk_by_id",
        lambda chunk_id: chunk_data.get(chunk_id),
    )

    filtered = orchestrator._apply_project_filter(
        [("chunk-a", 0.1), ("chunk-b", 0.2), ("chunk-none", 0.3)],
        project_filter=["project-b"],
    )

    assert filtered == [("chunk-b", 0.2)]


def test_project_filter_is_noop_when_empty(monkeypatch) -> None:
    orchestrator = _orchestrator(detected_project=None)
    monkeypatch.setattr(
        orchestrator._vector,
        "get_chunk_by_id",
        lambda chunk_id: {"metadata": {"project_id": "project-a"}},
    )

    fused = [("chunk-a", 0.05)]

    assert orchestrator._apply_project_filter(fused, project_filter=[]) == fused