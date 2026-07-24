"""PipelineExecutor: walks a `PipelineSpec` against a stage registry.

Turns `PipelineSpec` from inert data into something that actually runs:
resolve each named `StageSpec` via a registry of factories, build the
concrete stage (injecting any orchestrator-bound callables via
`StageDeps`), and run it. This is the single point where "which stage
runs here, with what config" is data (the spec + registry), not code --
adding/removing a stage becomes a spec edit.

Handles both sync `SearchStage` and coroutine `AsyncSearchStage`
uniformly: `run_stage` is a coroutine that awaits the stage's result
only if it is awaitable, so callers (already async, e.g.
`SearchOrchestrator.query`) don't need to special-case which stages are
I/O-bound.
"""

from __future__ import annotations

import inspect

from searchkernel.pipeline.registry import StageDeps, StageFactory
from searchkernel.pipeline.spec import PipelineSpec
from searchkernel.pipeline.stage import SearchContext


class UnknownStageError(KeyError):
    """A `PipelineSpec` named a stage absent from the registry."""


class PipelineExecutor:
    """Resolves and runs `PipelineSpec` stages against a stage registry."""

    def __init__(self, registry: dict[str, StageFactory]):
        self._registry = registry

    async def run_stage(
        self,
        name: str,
        config: dict,
        context: SearchContext,
        deps: StageDeps | None = None,
    ) -> SearchContext:
        try:
            factory = self._registry[name]
        except KeyError as exc:
            raise UnknownStageError(name) from exc

        result = factory(config, deps or StageDeps()).run(context)
        if inspect.isawaitable(result):
            return await result
        return result

    async def run(
        self,
        spec: PipelineSpec,
        context: SearchContext,
        deps: StageDeps | None = None,
    ) -> SearchContext:
        for stage_spec in spec.stages:
            context = await self.run_stage(
                stage_spec.name, stage_spec.config, context, deps
            )
        return context
