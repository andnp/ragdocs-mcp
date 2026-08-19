"""Instance-owned task runtime contracts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Any, Callable, Protocol

from mcp_markdown_ragdocs.coordination.queue import QueueRuntime
from mcp_markdown_ragdocs.coordination.task_submission import TaskSubmissionPort


class RegisteredTaskFunction(Protocol):
    __name__: str

    def __call__(self, *args: object, **kwargs: object) -> object: ...


class RegisteredTaskHandle(Protocol):
    func: RegisteredTaskFunction

    def __call__(self, *args: object, **kwargs: object) -> object: ...


@dataclass(frozen=True)
class TaskRuntime:
    """Own one queue, its submission port, and registered task handles."""

    queue_runtime: QueueRuntime
    submission: TaskSubmissionPort
    task_handles: Mapping[str, RegisteredTaskHandle] = field(default_factory=dict)
    gdrive_task_handles: Mapping[str, object] = field(default_factory=dict)
