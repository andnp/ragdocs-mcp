from dataclasses import replace

import pytest

from searchkernel.pipeline.executor import PipelineExecutor, UnknownStageError
from searchkernel.pipeline.spec import PipelineSpec, StageSpec
from searchkernel.pipeline.stage import SearchContext


class _AppendStage:
    def __init__(self, suffix: str):
        self.name = f"append_{suffix}"
        self._suffix = suffix

    def run(self, context: SearchContext) -> SearchContext:
        return replace(context, query=context.query + self._suffix)


class _AsyncAppendStage:
    def __init__(self, suffix: str):
        self.name = f"async_append_{suffix}"
        self._suffix = suffix

    async def run(self, context: SearchContext) -> SearchContext:
        return replace(context, query=context.query + self._suffix)


def _registry():
    return {
        "append": lambda config, deps: _AppendStage(config.get("suffix", "?")),
        "async_append": lambda config, deps: _AsyncAppendStage(
            config.get("suffix", "?")
        ),
    }


@pytest.mark.asyncio
async def test_run_stage_runs_a_sync_stage():
    executor = PipelineExecutor(_registry())

    result = await executor.run_stage(
        "append", {"suffix": "-a"}, SearchContext(query="q")
    )

    assert result.query == "q-a"


@pytest.mark.asyncio
async def test_run_stage_awaits_an_async_stage():
    executor = PipelineExecutor(_registry())

    result = await executor.run_stage(
        "async_append", {"suffix": "-b"}, SearchContext(query="q")
    )

    assert result.query == "q-b"


@pytest.mark.asyncio
async def test_run_stage_raises_for_an_unregistered_stage_name():
    executor = PipelineExecutor(_registry())

    with pytest.raises(UnknownStageError):
        await executor.run_stage("missing", {}, SearchContext(query="q"))


@pytest.mark.asyncio
async def test_run_walks_a_spec_threading_context_through_mixed_sync_async_stages():
    executor = PipelineExecutor(_registry())
    spec = PipelineSpec(
        name="test",
        stages=(
            StageSpec(name="append", config={"suffix": "-1"}),
            StageSpec(name="async_append", config={"suffix": "-2"}),
            StageSpec(name="append", config={"suffix": "-3"}),
        ),
    )

    result = await executor.run(spec, SearchContext(query="q"))

    assert result.query == "q-1-2-3"


@pytest.mark.asyncio
async def test_run_does_not_mutate_the_input_context():
    executor = PipelineExecutor(_registry())
    spec = PipelineSpec(
        name="test", stages=(StageSpec(name="append", config={"suffix": "-1"}),)
    )
    context = SearchContext(query="q")

    await executor.run(spec, context)

    assert context.query == "q"
