"""Measure query hydration for document and explicit Git source scopes."""

from __future__ import annotations

import statistics
import time
from pathlib import Path
from typing import TypedDict

import pytest
from mcp_markdown_ragdocs.config import ChunkingConfig, Config, IndexingConfig, LLMConfig, SearchConfig
from mcp_markdown_ragdocs.models import ChunkResult
from tests.integration._canonical import make_record, make_record_index_manager, make_search_adapter
from tests.performance.test_query_latency import create_benchmark_corpus


pytestmark = pytest.mark.performance


class BenchmarkMeasurement(TypedDict):
    latency_ms: float
    samples_ms: list[float]
    candidate_counts: dict[str, object]
    candidate_count: int
    stage_timings_ms: dict[str, object]
    results: list[ChunkResult]


@pytest.fixture
def config(tmp_path: Path) -> Config:
    """Build an isolated, offline configuration for the mixed corpus."""
    return Config(
        indexing=IndexingConfig(
            documents_path=str(tmp_path / "docs"),
            index_path=str(tmp_path / "indices"),
        ),
        search=SearchConfig(semantic_weight=1.0, keyword_weight=1.0),
        llm=LLMConfig(embedding_model="local"),
        chunking=ChunkingConfig(),
    )


@pytest.fixture
def indexed_mixed_corpus(config: Config):
    """Seed addressable Markdown documents beside pathless Git records."""
    documents_path = Path(config.indexing.documents_path)
    documents_path.mkdir()
    create_benchmark_corpus(documents_path, num_docs=16)
    manager = make_record_index_manager(config)
    for document in sorted(documents_path.glob("*.md")):
        assert manager.index_document(str(document))

    git_records = [
        make_record(
            f"git:commit-{index:03d}",
            "Testing performance deployment uses deterministic release checklists. "
            f"This Git change documents rollout signal {index:03d}.",
            source_kind="git_commit",
            metadata={
                "commit_id": f"commit-{index:03d}",
                "source_kind": "git_commit",
                "chunk_section": "summary",
            },
        )
        for index in range(16)
    ]
    assert manager.index_records(git_records)
    return manager, make_search_adapter(manager, config)


async def _measure(
    adapter, *, source_filter: list[str] | None
) -> BenchmarkMeasurement:
    query = "testing performance deployment"
    await adapter.query(query, top_k=8, top_n=8, source_filter=source_filter)
    samples: list[float] = []
    diagnostics: list[dict[str, object]] = []
    result_sets = []
    for _ in range(3):
        started = time.perf_counter()
        results, _, _ = await adapter.query(
            query, top_k=8, top_n=8, source_filter=source_filter
        )
        samples.append((time.perf_counter() - started) * 1000)
        diagnostics.append(dict(adapter.last_query_execution_stats))
        result_sets.append(results)
    candidate_count_value = diagnostics[-1].get("candidate_count", 0)
    candidate_count = (
        candidate_count_value if isinstance(candidate_count_value, int) else 0
    )
    candidate_counts_value = diagnostics[-1].get("candidate_counts", {})
    candidate_counts = (
        {str(key): value for key, value in candidate_counts_value.items()}
        if isinstance(candidate_counts_value, dict)
        else {}
    )
    stage_timings_value = diagnostics[-1].get("stage_timings_ms", {})
    stage_timings = (
        {str(key): value for key, value in stage_timings_value.items()}
        if isinstance(stage_timings_value, dict)
        else {}
    )
    return {
        "latency_ms": statistics.median(samples),
        "samples_ms": samples,
        "candidate_counts": candidate_counts or {"total": candidate_count},
        "candidate_count": candidate_count,
        "stage_timings_ms": stage_timings or {"query_total": samples[-1]},
        "results": result_sets[-1],
    }


@pytest.mark.asyncio
async def test_document_and_git_scope_hydration_benchmark(indexed_mixed_corpus):
    """Compare scoped retrieval while preserving source and result contracts."""
    _manager, adapter = indexed_mixed_corpus

    document = await _measure(adapter, source_filter=None)
    git = await _measure(adapter, source_filter=["git_commit"])
    document_results = document["results"]
    git_results = git["results"]

    assert document_results
    assert git_results
    assert all(result.metadata["source_kind"] == "note" for result in document_results)
    assert all(result.metadata["source_kind"] == "git_commit" for result in git_results)
    assert {result.doc_id for result in document_results}.isdisjoint(
        result.doc_id for result in git_results
    )
    for measurement in (document, git):
        assert isinstance(measurement["candidate_count"], int)
        assert measurement["candidate_count"] > 0
        assert isinstance(measurement["candidate_counts"], dict)
        assert measurement["stage_timings_ms"]
        print(
            f"scope={'git' if measurement is git else 'document'} "
            f"latency_ms={measurement['latency_ms']:.3f} "
            f"samples_ms={measurement['samples_ms']} "
            f"candidate_count={measurement['candidate_count']} "
            f"candidate_counts={measurement['candidate_counts']} "
            f"stage_timings_ms={measurement['stage_timings_ms']}"
        )
