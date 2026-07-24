from dataclasses import replace

import pytest

from searchkernel.pipeline.stage import AsyncSearchStage, SearchContext, SearchStage


class _UppercaseStage:
    name = "uppercase"

    def run(self, context: SearchContext) -> SearchContext:
        return replace(context, query=context.query.upper())


class _AsyncUppercaseStage:
    name = "async_uppercase"

    async def run(self, context: SearchContext) -> SearchContext:
        return replace(context, query=context.query.upper())


def test_search_context_defaults():
    context = SearchContext(query="hello")

    assert context.candidates == []
    assert context.strategy_results == {}
    assert context.metadata == {}


def test_stage_is_structurally_a_search_stage():
    stage = _UppercaseStage()

    assert isinstance(stage, SearchStage)


def test_stage_run_returns_new_context_without_mutating_input():
    context = SearchContext(query="hello")
    stage = _UppercaseStage()

    result = stage.run(context)

    assert result.query == "HELLO"
    assert context.query == "hello"


def test_async_stage_is_structurally_an_async_search_stage():
    stage = _AsyncUppercaseStage()

    assert isinstance(stage, AsyncSearchStage)


@pytest.mark.asyncio
async def test_async_stage_run_returns_new_context_without_mutating_input():
    context = SearchContext(query="hello")
    stage = _AsyncUppercaseStage()

    result = await stage.run(context)

    assert result.query == "HELLO"
    assert context.query == "hello"
