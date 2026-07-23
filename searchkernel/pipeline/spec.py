"""PipelineSpec: a declarative, pinned description of a pipeline's stages.

A `PipelineSpec` names an ordered sequence of stages plus per-stage config.
It is pure data: building an executor that walks a spec and runs the named
stages against a registry is follow-on work (see the W4a plan notes); for
now specs exist so a pipeline's shape can be described, versioned, and
diffed independently of the code that (today) still runs it inline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class StageSpec:
    """One named stage entry within a `PipelineSpec`."""

    name: str
    config: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PipelineSpec:
    """An ordered, pinned sequence of stages for a query or ingestion pipeline."""

    name: str
    stages: tuple[StageSpec, ...] = field(default_factory=tuple)

    def stage_names(self) -> tuple[str, ...]:
        return tuple(stage.name for stage in self.stages)
