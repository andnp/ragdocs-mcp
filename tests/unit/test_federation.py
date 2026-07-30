"""Unit tests for search_anything: fan-out timeouts + retrieve-then-rerank-once."""

import asyncio
from collections.abc import Iterable
from typing import Any

import pytest

from searchkernel.domain import ScoredRef
from searchkernel.ports import Reranker, SearchableSource
from searchkernel.runtime.federation import search_anything
from searchkernel.runtime.registry import SourceRegistry


class _FastSource:
    source_kind = "fast"

    async def search(
        self, query: str, k: int, filters: dict[str, Any] | None = None
    ) -> Iterable[ScoredRef]:
        return [
            ScoredRef(
                source_id="fast-1",
                score=0.9,
                source_kind="fast",
                metadata={"text": "fast source result"},
            )
        ]


class _SlowSource:
    source_kind = "slow"

    async def search(
        self, query: str, k: int, filters: dict[str, Any] | None = None
    ) -> Iterable[ScoredRef]:
        await asyncio.sleep(1.0)
        return [
            ScoredRef(
                source_id="slow-1",
                score=0.9,
                source_kind="slow",
                metadata={"text": "slow source result"},
            )
        ]


class _FailingSource:
    source_kind = "failing"

    async def search(
        self, query: str, k: int, filters: dict[str, Any] | None = None
    ) -> Iterable[ScoredRef]:
        raise RuntimeError("source is down")


class _StubReranker:
    """Deterministic reranker: score = position of the text in a fixed table."""

    model_name = "stub-reranker"

    def __init__(self, scores_by_text: dict[str, float]):
        self._scores_by_text = scores_by_text

    def rerank(self, query: str, documents: list[str]) -> list[float]:
        return [self._scores_by_text.get(doc, 0.0) for doc in documents]


def _identity_reranker() -> _StubReranker:
    return _StubReranker({})


@pytest.mark.asyncio
async def test_slow_source_times_out_others_still_return():
    registry = SourceRegistry()
    registry.register(_FastSource())
    registry.register(_SlowSource())
    reranker = _StubReranker({"fast source result": 1.0})

    results = await search_anything(
        "query",
        registry=registry,
        reranker=reranker,
        per_source_timeout_s=0.05,
    )

    assert [r.source_id for r in results] == ["fast-1"]


@pytest.mark.asyncio
async def test_failing_source_yields_no_results_others_still_return():
    registry = SourceRegistry()
    registry.register(_FastSource())
    registry.register(_FailingSource())
    reranker = _StubReranker({"fast source result": 1.0})

    results = await search_anything(
        "query",
        registry=registry,
        reranker=reranker,
    )

    assert [r.source_id for r in results] == ["fast-1"]


@pytest.mark.asyncio
async def test_no_sources_registered_returns_empty():
    registry = SourceRegistry()

    results = await search_anything(
        "query",
        registry=registry,
        reranker=_identity_reranker(),
    )

    assert results == []


@pytest.mark.asyncio
async def test_top_n_truncates_the_fused_list():
    registry = SourceRegistry()
    registry.register(_FastSource())
    reranker = _StubReranker({"fast source result": 1.0})

    results = await search_anything(
        "query",
        registry=registry,
        reranker=reranker,
        top_n=0,
    )

    assert results == []


@pytest.mark.asyncio
async def test_sources_param_restricts_fan_out_to_named_sources():
    registry = SourceRegistry()
    registry.register(_FastSource())
    registry.register(_FailingSource())
    reranker = _StubReranker({"fast source result": 1.0})

    results = await search_anything(
        "query",
        registry=registry,
        reranker=reranker,
        sources=["fast"],
    )

    assert [r.source_id for r in results] == ["fast-1"]


def test_reranker_protocol_conformance_of_stub():
    assert isinstance(_StubReranker({}), Reranker)


def test_searchable_source_protocol_conformance_of_stubs():
    assert isinstance(_FastSource(), SearchableSource)
    assert isinstance(_SlowSource(), SearchableSource)
    assert isinstance(_FailingSource(), SearchableSource)


@pytest.mark.asyncio
async def test_rerank_once_reorders_by_reranker_not_source_score():
    """The merged set is ordered by the single rerank pass, not each source's
    own score -- a source's high self-reported score should not win if the
    reranker judges it less relevant than a lower-scored candidate."""
    registry = SourceRegistry()

    class _SourceA:
        source_kind = "a"

        async def search(self, query, k, filters=None):
            return [
                ScoredRef(
                    source_id="a-1",
                    score=0.99,  # highest self-reported score
                    source_kind="a",
                    metadata={"text": "irrelevant filler text"},
                )
            ]

    class _SourceB:
        source_kind = "b"

        async def search(self, query, k, filters=None):
            return [
                ScoredRef(
                    source_id="b-1",
                    score=0.1,  # lowest self-reported score
                    source_kind="b",
                    metadata={"text": "the most relevant answer"},
                )
            ]

    registry.register(_SourceA())
    registry.register(_SourceB())
    reranker = _StubReranker(
        {
            "irrelevant filler text": 0.05,
            "the most relevant answer": 0.95,
        }
    )

    results = await search_anything(
        "query",
        registry=registry,
        reranker=reranker,
    )

    assert [r.source_id for r in results] == ["b-1", "a-1"]
    assert results[0].score == 0.95
    assert results[0].metadata["source_score"] == 0.1
